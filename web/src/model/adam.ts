/** Adam with the exact update rule of `torch.optim.Adam` (no weight decay, no amsgrad). */
export class Adam {
  readonly m: Float64Array;
  readonly v: Float64Array;
  t = 0;

  constructor(
    size: number,
    public beta1 = 0.9,
    public beta2 = 0.999,
    public eps = 1e-8,
  ) {
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
  }

  reset(): void {
    this.m.fill(0);
    this.v.fill(0);
    this.t = 0;
  }

  step(params: Float64Array, grad: Float64Array, lr: number): void {
    const { m, v, beta1, beta2, eps } = this;
    this.t += 1;
    const bc1 = 1 - beta1 ** this.t;
    const bc2Sqrt = Math.sqrt(1 - beta2 ** this.t);
    const stepSize = lr / bc1;
    for (let i = 0; i < params.length; i++) {
      const g = grad[i];
      m[i] = beta1 * m[i] + (1 - beta1) * g;
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;
      params[i] -= (stepSize * m[i]) / (Math.sqrt(v[i]) / bc2Sqrt + eps);
    }
  }

  snapshot(): { m: Float64Array; v: Float64Array; t: number } {
    return { m: this.m.slice(), v: this.v.slice(), t: this.t };
  }

  restore(s: { m: Float64Array; v: Float64Array; t: number }): void {
    this.m.set(s.m);
    this.v.set(s.v);
    this.t = s.t;
  }
}
