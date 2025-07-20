from cosmos_predict2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler
from statistics import NormalDist
import numpy as np



gaussian_dist = NormalDist(mu=0.0, sigma=1.0)
cdf_vals = np.clip(np.linspace(0, 1.0, 17), 1e-8, 1 - 1e-8)
samples_interval_gaussian = [gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]
sigmas = np.exp(samples_interval_gaussian)


scheduler = RectifiedFlowAB2Scheduler()
scheduler.set_timesteps(16)

samples = scheduler.sample_sigma(10000)
interval_indices = np.searchsorted(sigmas, samples.cpu(), side='right') - 1