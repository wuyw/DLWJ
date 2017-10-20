package org.dl.util;

import java.util.Random;

public final class GaussianDistribution {
	private final double mean;
	private final double variance;
	private final Random random;

	public GaussianDistribution(double mean, double variance, Random random) {
		if (variance < 0.0) {
			throw new IllegalArgumentException("Variance must be non-negative value.");
		}

		this.mean = mean;
		this.variance = variance;

		if (random == null) {
			random = new Random();
		}
		this.random = random;
	}

	public double random() {
		double r = 0.0;
		while (r == 0.0) {
			r = random.nextDouble();
		}

		double c = Math.sqrt(-2.0 * Math.log(r));

		if (random.nextDouble() < 0.5) {
			return c * Math.sin(2.0 * Math.PI * random.nextDouble()) * variance + mean;
		}
		return c * Math.cos(2.0 * Math.PI * random.nextDouble()) * variance + mean;
	}
}