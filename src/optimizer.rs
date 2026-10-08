//! Stateful Adam, independent of tape node identities.
use crate::tensor::Tensor;
use std::collections::HashMap;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}
impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}
impl AdamConfig {
    pub fn validate(self) -> Result<(), String> {
        if !(0.0..1.0).contains(&self.beta1)
            || !(0.0..1.0).contains(&self.beta2)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
        {
            return Err("Adam requires beta1/beta2 in [0,1) and positive finite epsilon".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Moments {
    pub step: u64,
    pub first: Vec<f64>,
    pub second: Vec<f64>,
}
#[derive(Clone, Debug, Default)]
pub struct Adam {
    pub config: Option<AdamConfig>,
    pub states: HashMap<usize, Moments>,
}
impl Adam {
    /// Validate every candidate before committing any optimizer state.
    pub fn update(
        &mut self,
        parameters: &[(usize, Tensor, Vec<f64>)],
        rate: f64,
        config: AdamConfig,
    ) -> Result<Vec<(usize, Tensor)>, String> {
        config.validate()?;
        if !rate.is_finite() || rate <= 0.0 {
            return Err("learning rate must be positive and finite".into());
        }
        if self.config.is_some_and(|old| old != config) {
            return Err("Adam hyperparameters cannot change after state initialization".into());
        }
        let mut updates = vec![];
        let mut states = vec![];
        for (slot, value, gradient) in parameters {
            if gradient.len() != value.data.len() {
                return Err("Adam gradient size mismatch".into());
            }
            let mut state = self.states.get(slot).cloned().unwrap_or_else(|| Moments {
                step: 0,
                first: vec![0.0; gradient.len()],
                second: vec![0.0; gradient.len()],
            });
            if state.first.len() != gradient.len() || state.second.len() != gradient.len() {
                return Err("Adam state size mismatch".into());
            }
            state.step = state
                .step
                .checked_add(1)
                .ok_or("Adam step counter overflow")?;
            let correction1 = 1.0 - config.beta1.powf(state.step as f64);
            let correction2 = 1.0 - config.beta2.powf(state.step as f64);
            let mut data = value.data.clone();
            for (((x, m), v), &g) in data
                .iter_mut()
                .zip(&mut state.first)
                .zip(&mut state.second)
                .zip(gradient)
            {
                *m = config.beta1 * *m + (1.0 - config.beta1) * g;
                *v = config.beta2 * *v + (1.0 - config.beta2) * g * g;
                if !m.is_finite() || !v.is_finite() || *v < 0.0 {
                    return Err("non-finite Adam moments".into());
                }
                *x -= rate * (*m / correction1) / ((*v / correction2).sqrt() + config.epsilon);
            }
            updates.push((*slot, Tensor::new(value.shape.clone(), data)?));
            states.push((*slot, state));
        }
        self.config = Some(config);
        for (slot, state) in states {
            self.states.insert(slot, state);
        }
        Ok(updates)
    }
}

/// A normalized clipping transform: divide each gradient by the first component,
/// then multiply by the second. This avoids overflow and scale-factor underflow.
/// Parameter-order iteration is deterministic in the VM.
pub fn clip_transform<'a>(
    gradients: impl Iterator<Item = &'a [f64]> + Clone,
    maximum: f64,
) -> Result<Option<(f64, f64)>, String> {
    if !maximum.is_finite() || maximum <= 0.0 {
        return Err("clip_grad_norm maximum must be positive and finite".into());
    }
    let mut largest = 0.0f64;
    for &value in gradients.clone().flatten() {
        if !value.is_finite() {
            return Err("cannot clip non-finite gradients".into());
        }
        largest = largest.max(value.abs());
    }
    if largest == 0.0 {
        return Ok(None);
    }
    let norm_scaled = gradients
        .flatten()
        .map(|value| (value / largest).powi(2))
        .sum::<f64>()
        .sqrt();
    if largest <= maximum / norm_scaled {
        return Ok(None);
    }
    Ok(Some((largest, maximum / norm_scaled)))
}
