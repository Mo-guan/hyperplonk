// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Prover subroutines for a SumCheck protocol.

use super::SumCheckProver;
use crate::poly_iop::{
    errors::PolyIOPErrors,
    structs::{IOPProverMessage, IOPProverState},
};
use arithmetic::{compute_eq_w_x, eq_prefix, fix_variables, VirtualPolynomial};
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_std::{cfg_into_iter, end_timer, start_timer, vec::Vec};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use std::sync::Arc;
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

impl<F: PrimeField> SumCheckProver<F> for IOPProverState<F> {
    type VirtualPolynomial = VirtualPolynomial<F>;
    type ProverMessage = IOPProverMessage<F>;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    fn prover_init(polynomial: &Self::VirtualPolynomial) -> Result<Self, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prover init");
        if polynomial.aux_info.num_variables == 0 {
            return Err(PolyIOPErrors::InvalidParameters(
                "Attempt to prove a constant.".to_string(),
            ));
        }
        end_timer!(start);

        Ok(Self {
            challenges: Vec::with_capacity(polynomial.aux_info.num_variables),
            round: 0,
            poly: polynomial.clone(),
            extrapolation_aux: (1..polynomial.aux_info.max_degree + 1)
                .map(|degree| {
                    let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
                    let weights = barycentric_weights(&points);
                    (points, weights)
                })
                .collect(),
        })
    }

    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    fn prove_round_and_update_state(
        &mut self,
        challenge: &Option<F>,
    ) -> Result<Self::ProverMessage, PolyIOPErrors> {
        // let start =
        //     start_timer!(|| format!("sum check prove {}-th round and update state",
        // self.round));

        if self.round >= self.poly.aux_info.num_variables {
            return Err(PolyIOPErrors::InvalidProver(
                "Prover is not active".to_string(),
            ));
        }

        // let fix_argument = start_timer!(|| "fix argument");

        // Step 1:
        // fix argument and evaluate f(x) over x_m = r; where r is the challenge
        // for the current round, and m is the round number, indexed from 1
        //
        // i.e.:
        // at round m <= n, for each mle g(x_1, ... x_n) within the flattened_mle
        // which has already been evaluated to
        //
        //    g(r_1, ..., r_{m-1}, x_m ... x_n)
        //
        // eval g over r_m, and mutate g to g(r_1, ... r_m,, x_{m+1}... x_n)
        let mut flattened_ml_extensions: Vec<DenseMultilinearExtension<F>> = self
            .poly
            .flattened_ml_extensions
            .par_iter()
            .map(|x| x.as_ref().clone())
            .collect();

        if let Some(chal) = challenge {
            if self.round == 0 {
                return Err(PolyIOPErrors::InvalidProver(
                    "first round should be prover first.".to_string(),
                ));
            }
            self.challenges.push(*chal);

            let r = self.challenges[self.round - 1];
            #[cfg(feature = "parallel")]
            flattened_ml_extensions
                .par_iter_mut()
                .for_each(|mle| *mle = fix_variables(mle, &[r]));
            #[cfg(not(feature = "parallel"))]
            flattened_ml_extensions
                .iter_mut()
                .for_each(|mle| *mle = fix_variables(mle, &[r]));
        } else if self.round > 0 {
            return Err(PolyIOPErrors::InvalidProver(
                "verifier message is empty".to_string(),
            ));
        }
        // end_timer!(fix_argument);

        self.round += 1;

        let products_list = self.poly.products.clone();
        let mut products_sum = vec![F::zero(); self.poly.aux_info.max_degree + 1];

        // Step 2: generate sum for the partial evaluated polynomial:
        // f(r_1, ... r_m,, x_{m+1}... x_n)

        products_list.iter().for_each(|(coefficient, products)| {
            let mut sum = cfg_into_iter!(0..1 << (self.poly.aux_info.num_variables - self.round))
                .fold(
                    || {
                        (
                            vec![(F::zero(), F::zero()); products.len()],
                            vec![F::zero(); products.len() + 1],
                        )
                    },
                    |(mut buf, mut acc), b| {
                        buf.iter_mut()
                            .zip(products.iter())
                            .for_each(|((eval, step), f)| {
                                let table = &flattened_ml_extensions[*f];
                                *eval = table[b << 1];
                                *step = table[(b << 1) + 1] - table[b << 1];
                            });
                        acc[0] += buf.iter().map(|(eval, _)| eval).product::<F>();
                        acc[1..].iter_mut().for_each(|acc| {
                            buf.iter_mut().for_each(|(eval, step)| *eval += step as &_);
                            *acc += buf.iter().map(|(eval, _)| eval).product::<F>();
                        });
                        (buf, acc)
                    },
                )
                .map(|(_, partial)| partial)
                .reduce(
                    || vec![F::zero(); products.len() + 1],
                    |mut sum, partial| {
                        sum.iter_mut()
                            .zip(partial.iter())
                            .for_each(|(sum, partial)| *sum += partial);
                        sum
                    },
                );
            sum.iter_mut().for_each(|sum| *sum *= coefficient);
            let extraploation = cfg_into_iter!(0..self.poly.aux_info.max_degree - products.len())
                .map(|i| {
                    let (points, weights) = &self.extrapolation_aux[products.len() - 1];
                    let at = F::from((products.len() + 1 + i) as u64);
                    extrapolate(points, weights, &sum, &at)
                })
                .collect::<Vec<_>>();
            products_sum
                .iter_mut()
                .zip(sum.iter().chain(extraploation.iter()))
                .for_each(|(products_sum, sum)| *products_sum += sum);
        });
        self.poly.flattened_ml_extensions = flattened_ml_extensions
            .par_iter()
            .map(|x| Arc::new(x.clone()))
            .collect();

        Ok(IOPProverMessage {
            evaluations: products_sum,
        })
    }

    fn prove_round_and_update_state_sqrt_space(
        &mut self,
        challenge: &Option<F>,
        r: &Vec<F>,
    ) -> Result<Self::ProverMessage, PolyIOPErrors> {
        println!(
            "----------------------- round {} -----------------------",
            self.round
        );

        if self.round >= self.poly.aux_info.num_variables {
            return Err(PolyIOPErrors::InvalidProver(
                "Prover is not active".to_string(),
            ));
        }

        let mle_tables: Vec<DenseMultilinearExtension<F>> = self
            .poly
            .flattened_ml_extensions
            .par_iter()
            .map(|x| x.as_ref().clone())
            .collect();

        if let Some(chal) = challenge {
            if self.round == 0 {
                return Err(PolyIOPErrors::InvalidProver(
                    "first round should be prover first.".to_string(),
                ));
            }
            self.challenges.push(*chal);
        } else if self.round > 0 {
            return Err(PolyIOPErrors::InvalidProver(
                "verifier message is empty".to_string(),
            ));
        }

        self.round += 1;

        let products = self.poly.products.clone();
        let max_deg = self.poly.aux_info.max_degree;

        let mut products_sum = vec![F::zero(); max_deg + 2];

        let l = self.poly.aux_info.num_variables;
        let left_bits = l / 2;
        let right_bits = l - left_bits;

        let round_idx = self.round;
        let (eq_right, eq_left): (Vec<F>, Option<Vec<F>>) = if round_idx < left_bits {
            let w_r = &r[left_bits..l];
            let w_l_tail = &r[round_idx..left_bits];
            (compute_eq_w_x(w_r), Some(compute_eq_w_x(w_l_tail)))
        } else {
            let w_r_tail = &r[round_idx..l];
            (
                if w_r_tail.is_empty() {
                    vec![F::one()]
                } else {
                    compute_eq_w_x(w_r_tail)
                },
                None,
            )
        };

        let (bits_r, bits_l) = if round_idx < left_bits {
            (right_bits, left_bits - round_idx)
        } else {
            (l - round_idx, 0)
        };

        let mut accs_per_product: Vec<Vec<F>> = products
            .iter()
            .map(|(_coef, factors)| vec![F::zero(); factors.len() + 1])
            .collect();

        for xr in 0..(1 << bits_r) {
            let wr = eq_right[xr];
            for xl in 0..(1 << bits_l) {
                let x = (xr << bits_l) | xl;
                let mut sk_cache: HashMap<usize, Vec<F>> = HashMap::new();

                let weights_span_opt = if self.round > 1 {
                    Some((compute_eq_w_x(&self.challenges), 1 << (self.round - 1)))
                } else {
                    None
                };

                let wl = if self.round < left_bits {
                    eq_left.as_ref().unwrap()[xl]
                } else {
                    F::one()
                };
                let w = wr * wl;

                for (j, (coef, factors)) in products.iter().enumerate() {
                    let d = factors.len() + 1;
                    let mut prod_vals = vec![*coef; d];

                    for &fi in factors.iter() {
                        let need_len = d;
                        let sk_slice: &[F] = if let Some(existing) = sk_cache.get(&fi) {
                            if existing.len() >= need_len {
                                &existing[..need_len]
                            } else {
                                let mut sk = vec![F::zero(); need_len];
                                let table = &mle_tables[fi];
                                if self.round == 1 {
                                    sk[0] = table[x << 1];
                                    sk[1] = table[(x << 1) + 1];
                                    let diff = sk[1] - sk[0];
                                    for y in 2..need_len {
                                        sk[y] = sk[y - 1] + diff;
                                    }
                                } else {
                                    let (weights, span) = weights_span_opt.as_ref().unwrap();
                                    for b in 0..*span {
                                        let mut p = vec![F::zero(); need_len];
                                        p[0] = table[((x << 1) << (self.round - 1)) + b];
                                        p[1] = table[((1 + (x << 1)) << (self.round - 1)) + b];
                                        let diff = p[1] - p[0];
                                        for y in 2..need_len {
                                            p[y] = p[y - 1] + diff;
                                        }
                                        let wb = weights[b];
                                        for y in 0..need_len {
                                            sk[y] += p[y] * wb;
                                        }
                                    }
                                }
                                sk_cache.insert(fi, sk);
                                &sk_cache.get(&fi).unwrap()[..need_len]
                            }
                        } else {
                            let mut sk = vec![F::zero(); need_len];
                            let table = &mle_tables[fi];
                            if self.round == 1 {
                                sk[0] = table[x << 1];
                                sk[1] = table[(x << 1) + 1];
                                let diff = sk[1] - sk[0];
                                for y in 2..need_len {
                                    sk[y] = sk[y - 1] + diff;
                                }
                            } else {
                                let (weights, span) = weights_span_opt.as_ref().unwrap();
                                for b in 0..*span {
                                    let mut p = vec![F::zero(); need_len];
                                    p[0] = table[((x << 1) << (self.round - 1)) + b];
                                    p[1] = table[((1 + (x << 1)) << (self.round - 1)) + b];
                                    let diff = p[1] - p[0];
                                    for y in 2..need_len {
                                        p[y] = p[y - 1] + diff;
                                    }
                                    let wb = weights[b];
                                    for y in 0..need_len {
                                        sk[y] += p[y] * wb;
                                    }
                                }
                            }
                            sk_cache.insert(fi, sk);
                            &sk_cache.get(&fi).unwrap()[..need_len]
                        };

                        for y in 0..d {
                            prod_vals[y] *= sk_slice[y];
                        }
                    }

                    for y in 0..d {
                        accs_per_product[j][y] += prod_vals[y] * w;
                    }
                }
            }
        }

        for (j, (_coef, factors)) in products.iter().enumerate() {
            let k = factors.len();
            let d = k + 1;
            for y in 0..d {
                products_sum[y] += accs_per_product[j][y];
            }
            if k < max_deg + 1 {
                let (points, weights) = &self.extrapolation_aux[k.saturating_sub(1)];
                for t in 0..(max_deg + 1 - k) {
                    let at = F::from((k + 1 + t) as u64);
                    let val = extrapolate(points, weights, &accs_per_product[j], &at);
                    products_sum[k + 1 + t] += val;
                }
            }
        }

        let alpha = if self.round > 1 {
            eq_prefix(
                &r[0..self.round - 1],
                &self.challenges[0..self.round - 1],
                self.round - 1,
            )
        } else {
            F::one()
        };
        let two = F::from(2u64);
        let ri = r[self.round - 1];
        for (i, val) in products_sum.iter_mut().enumerate().take(max_deg + 2) {
            let y = F::from(i as u64);
            let li = alpha * ((F::one() - ri) + (two * ri - F::one()) * y);
            *val *= li;
        }

        Ok(IOPProverMessage {
            evaluations: products_sum,
        })
    }

    fn prove_round_and_update_state_with_compute_opti(
        &mut self,
        challenge: &Option<F>,
    ) -> Result<Self::ProverMessage, PolyIOPErrors> {
        if self.round >= self.poly.aux_info.num_variables {
            return Err(PolyIOPErrors::InvalidProver(
                "Prover is not active".to_string(),
            ));
        }

        let mut flattened_ml_extensions: Vec<DenseMultilinearExtension<F>> = self
            .poly
            .flattened_ml_extensions
            .par_iter()
            .map(|x| x.as_ref().clone())
            .collect();

        if let Some(chal) = challenge {
            if self.round == 0 {
                return Err(PolyIOPErrors::InvalidProver(
                    "first round should be prover first.".to_string(),
                ));
            }
            self.challenges.push(*chal);
            let r = self.challenges[self.round - 1];

            #[cfg(feature = "parallel")]
            flattened_ml_extensions
                .par_iter_mut()
                .for_each(|mle| *mle = fix_variables(mle, &[r]));
            #[cfg(not(feature = "parallel"))]
            flattened_ml_extensions
                .iter_mut()
                .for_each(|mle| *mle = fix_variables(mle, &[r]));
        } else if self.round > 0 {
            return Err(PolyIOPErrors::InvalidProver(
                "verifier message is empty".to_string(),
            ));
        }

        self.round += 1;

        let products_list = self.poly.products.clone();
        let max_degree = self.poly.aux_info.max_degree;
        let num_vars = self.poly.aux_info.num_variables;
        let num_b: usize = 1 << (num_vars - self.round);
        let num_mle = flattened_ml_extensions.len();

        #[cfg(feature = "parallel")]
        let (eval0, step): (Vec<Vec<F>>, Vec<Vec<F>>) = {
            use rayon::prelude::*;
            let mut e = vec![vec![F::zero(); num_b]; num_mle];
            let mut s = vec![vec![F::zero(); num_b]; num_mle];
            e.par_iter_mut()
                .zip(s.par_iter_mut())
                .enumerate()
                .for_each(|(f, (e_row, s_row))| {
                    let table = &flattened_ml_extensions[f];
                    for b in 0..num_b {
                        let a0 = table[b << 1];
                        let a1 = table[(b << 1) + 1];
                        e_row[b] = a0;
                        s_row[b] = a1 - a0;
                    }
                });
            (e, s)
        };
        #[cfg(not(feature = "parallel"))]
        let (eval0, step): (Vec<Vec<F>>, Vec<Vec<F>>) = {
            let mut e = vec![vec![F::zero(); num_b]; num_mle];
            let mut s = vec![vec![F::zero(); num_b]; num_mle];
            for f in 0..num_mle {
                let table = &flattened_ml_extensions[f];
                for b in 0..num_b {
                    let a0 = table[b << 1];
                    let a1 = table[(b << 1) + 1];
                    e[f][b] = a0;
                    s[f][b] = a1 - a0;
                }
            }
            (e, s)
        };

        let num_mle = flattened_ml_extensions.len();
        let mut max_y_need_for_mle = vec![0usize; num_mle];
        for (_coef, prods) in products_list.iter() {
            let k = prods.len();
            for &f in prods.iter() {
                if k > max_y_need_for_mle[f] {
                    max_y_need_for_mle[f] = k;
                }
            }
        }

        let mut accs_per_product: Vec<Vec<F>> = products_list
            .iter()
            .map(|(_, prods)| vec![F::zero(); prods.len() + 1])
            .collect();

        let mut cur_vals: Vec<Vec<F>> = eval0.clone();

        for y in 0..=max_degree {
            if y > 0 {
                #[cfg(feature = "parallel")]
                use rayon::prelude::*;
                cur_vals
                    .par_iter_mut()
                    .zip(step.par_iter())
                    .zip(max_y_need_for_mle.par_iter())
                    .for_each(|((cur_row, step_row), &k_f)| {
                        if y <= k_f {
                            for b in 0..num_b {
                                cur_row[b] += step_row[b];
                            }
                        }
                    });
                #[cfg(not(feature = "parallel"))]
                for f in 0..num_mle {
                    if y <= max_y_need_for_mle[f] {
                        for b in 0..num_b {
                            cur_vals[f][b] += step[f][b];
                        }
                    }
                }
            }

            for (j, (_coef, prods)) in products_list.iter().enumerate() {
                let k = prods.len();
                if y > k {
                    continue;
                }

                let mut sum_y = F::zero();
                for b in 0..num_b {
                    let mut p = F::one();
                    for &f in prods.iter() {
                        p *= cur_vals[f][b];
                    }
                    sum_y += p;
                }
                accs_per_product[j][y] += sum_y;
            }
        }

        let mut products_sum = vec![F::zero(); max_degree + 1];

        for (j, (coefficient, prods)) in products_list.iter().enumerate() {
            let k = prods.len();
            for y in 0..=k {
                products_sum[y] += accs_per_product[j][y] * *coefficient;
            }

            if k < max_degree {
                let (points, weights) = &self.extrapolation_aux[k.saturating_sub(1)];
                for i in 0..(max_degree - k) {
                    let at = F::from((k + 1 + i) as u64);
                    let val = extrapolate(points, weights, &accs_per_product[j], &at);
                    products_sum[k + 1 + i] += val * *coefficient;
                }
            }
        }

        self.poly.flattened_ml_extensions = flattened_ml_extensions
            .par_iter()
            .map(|x| Arc::new(x.clone()))
            .collect();

        Ok(IOPProverMessage {
            evaluations: products_sum,
        })
    }
}

fn barycentric_weights<F: PrimeField>(points: &[F]) -> Vec<F> {
    let mut weights = points
        .iter()
        .enumerate()
        .map(|(j, point_j)| {
            points
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != j)
                .map(|(_i, point_i)| *point_j - point_i)
                .reduce(|acc, value| acc * value)
                .unwrap_or_else(F::one)
        })
        .collect::<Vec<_>>();
    batch_inversion(&mut weights);
    assert_eq!(
        weights.len(),
        points.len(),
        "barycentric weights length mismatch"
    );
    weights
}

fn extrapolate<F: PrimeField>(points: &[F], weights: &[F], evals: &[F], at: &F) -> F {
    let (coeffs, sum_inv) = {
        let mut coeffs = points.iter().map(|point| *at - point).collect::<Vec<_>>();
        batch_inversion(&mut coeffs);
        coeffs.iter_mut().zip(weights).for_each(|(coeff, weight)| {
            *coeff *= weight;
        });
        let sum_inv = coeffs.iter().sum::<F>().inverse().unwrap_or_default();
        (coeffs, sum_inv)
    };
    coeffs
        .iter()
        .zip(evals)
        .map(|(coeff, eval)| *coeff * eval)
        .sum::<F>()
        * sum_inv
}

#[cfg(test)]
mod tests_equivalence {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_poly::MultilinearExtension;
    use ark_std::test_rng;

    fn assert_mle_vec_eq<F: PrimeField>(
        a: &Vec<Arc<DenseMultilinearExtension<F>>>,
        b: &Vec<Arc<DenseMultilinearExtension<F>>>,
    ) {
        assert_eq!(a.len(), b.len(), "flattened MLE length mismatch");
        for (i, (aa, bb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                aa.num_vars(),
                bb.num_vars(),
                "MLE num_vars mismatch at {}",
                i
            );
            assert_eq!(
                aa.to_evaluations(),
                bb.to_evaluations(),
                "MLE evals mismatch at {}",
                i
            );
            println!("{}", aa.to_evaluations()[0]);
        }
    }

    fn run_equivalence_once(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        let (poly, _sum) =
            VirtualPolynomial::<Fr>::rand(nv, num_multiplicands_range, num_products, &mut rng)
                .expect("random VP");

        let r = (0..poly.aux_info.num_variables)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let f_hat = poly.build_f_hat(r.as_ref())?;

        let challenges = (0..poly.aux_info.num_variables)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        // println!("products_list: {:?}", poly.products);
        // let products_list = poly.products.clone();
        // for (_coef, prods) in products_list.iter() {
        //     for &f in prods.iter() {
        //         let mle = &poly.flattened_ml_extensions[f];
        //         println!(
        //             "mle[{}]: num_vars={}, evals={:?}",
        //             f,
        //             mle.num_vars(),
        //             mle.to_evaluations()
        //         );
        //         let a0 = mle.to_evaluations();
        //         let mut a1 = vec![Fr::zero(); 1 << (nv - 1)];
        //         println!(
        //             "a0[0] = {:?}, a0[1] = {:?}, a0[0] * a0[1] = {:?}",
        //             a0[0],
        //             a0[1],
        //             (a0[0] * a0[1])
        //         );
        //         a1[0] = a0[0] + (a0[1] - a0[0]) * challenges[0];
        //         a1[1] = a0[2] + (a0[3] - a0[2]) * challenges[0];
        //         println!("  a1: {:?}", a1);
        //     }
        // }

        let mut s_naive = IOPProverState::prover_init(&f_hat)?;
        let mut s_opti = IOPProverState::prover_init(&f_hat)?;
        let mut s_ss = IOPProverState::prover_init(&poly)?;

        for round in 0..poly.aux_info.num_variables {
            let mut challenge: Option<Fr> = None;
            if round > 0 {
                challenge = Some(challenges[round - 1]);
            }
            let msg_naive = IOPProverState::prove_round_and_update_state(&mut s_naive, &challenge)?;
            let msg_opti = IOPProverState::prove_round_and_update_state_with_compute_opti(
                &mut s_opti,
                &challenge,
            )?;
            let msg_ss =
                IOPProverState::prove_round_and_update_state_sqrt_space(&mut s_ss, &challenge, &r)?;

            assert_eq!(
                msg_naive.evaluations, msg_opti.evaluations,
                "prover messages not equal in this round",
            );
            println!("msg_naive.evaluations.len(): {}, msg_opti.evaluations.len(): {}, msg_ss.evaluations.len(): {}", msg_naive.evaluations.len(), msg_opti.evaluations.len(), msg_ss.evaluations.len());
            assert_eq!(
                msg_naive.evaluations, msg_ss.evaluations,
                "prover messages not equal in this round",
            );
        }

        s_naive.challenges.push(challenges.last().copied().unwrap());
        s_opti.challenges.push(challenges.last().copied().unwrap());
        s_ss.challenges.push(challenges.last().copied().unwrap());

        assert_eq!(s_naive.round, s_opti.round, "round mismatch");
        assert_eq!(s_naive.challenges, s_opti.challenges, "challenges mismatch");
        assert_eq!(
            s_naive.poly.products, s_opti.poly.products,
            "products layout mismatch",
        );
        assert_eq!(s_naive.round, s_ss.round, "round mismatch with sqrt space",);
        assert_eq!(
            s_naive.challenges, s_ss.challenges,
            "challenges mismatch with sqrt space",
        );

        // products layout should be different
        // assert_eq!(
        //     s_naive.poly.products, s_ss.poly.products,
        //     "products layout mismatch with sqrt space",
        // );
        assert_mle_vec_eq(
            &s_naive.poly.flattened_ml_extensions,
            &s_opti.poly.flattened_ml_extensions,
        );

        // mle should be different
        // assert_mle_vec_eq(
        //     &s_naive.poly.flattened_ml_extensions,
        //     &s_ss.poly.flattened_ml_extensions,
        // );

        println!("  passed one equivalence test with nv={nv}, num_multiplicands_range={num_multiplicands_range:?}, num_products={num_products}");
        Ok(())
    }

    #[test]
    fn test_prover_round_equivalence_small() -> Result<(), PolyIOPErrors> {
        run_equivalence_once(2, (2, 4), 2)?;
        run_equivalence_once(2, (2, 3), 1)
    }

    #[test]
    fn test_prover_round_equivalence_medium() -> Result<(), PolyIOPErrors> {
        run_equivalence_once(5, (2, 5), 4)?;
        run_equivalence_once(6, (3, 6), 5)
    }
}
