export type RegType = 'none' | 'l1' | 'l2';

/** Adds the L1/L2 penalty gradient to `grad` and returns the penalty (`_apply_regularization`). */
export function applyRegularization(
  params: Float64Array,
  grad: Float64Array,
  type: RegType,
  strength: number,
): number {
  let penalty = 0;
  if (type === 'l1') {
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      penalty += Math.abs(p);
      grad[i] += strength * Math.sign(p);
    }
  } else if (type === 'l2') {
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      penalty += p * p;
      grad[i] += 2 * strength * p;
    }
  }
  return strength * penalty;
}
