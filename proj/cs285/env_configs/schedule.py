import numpy as np

import yaml
class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()

def get_schedule_type(schedule_type: str):
        if schedule_type == "linear":
            return LinearSchedule
        elif schedule_type == "piecewise":
            return PiecewiseSchedule
        elif schedule_type == "constant":
            return ConstantSchedule
        elif schedule_type == "adaptive":
            return AdaptiveRewardBasedSchedule
        elif schedule_type == "sinusoidal":
            return SinusoidalSchedule
        else:
            raise ValueError("Invalid schedule type {}".format(schedule_type))
    
def get_schedule(exploration_schedule_file) -> Schedule:
    with open(exploration_schedule_file, "r") as f:
        exploration_kwargs = yaml.load(f, Loader=yaml.SafeLoader)
    schedule_type = exploration_kwargs["schedule_type"]
    schedule_type = get_schedule_type(schedule_type)
    exploration_kwargs.pop("schedule_type")
    return schedule_type(**exploration_kwargs)

class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t, reward):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class SinusoidalSchedule:
    
    def __init__(self, schedule_timesteps, period_frac, final_p, initial_p=1.0):
        """Sinusoidal interpolation between initial_p and final_p over"""
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.period = period_frac * schedule_timesteps
    
    def value(self, t, reward):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return (self.initial_p + fraction * (self.final_p - self.initial_p)) * (np.cos(2 * np.pi * t/self.period) ** 2)
    

class AdaptiveRewardBasedSchedule:
    
    def __init__(self, 
                 n=10000, 
                 p=0.005, 
                 alpha=0.99, 
                 threshold=0.15, 
                 eps_max=0.5):
        """
        Adaptive schedule based on reward.
        Parameters
        ----------
        n: int
            Number of steps to look back when computing the relative change 
            in reward.
        p: float
            Fraction of n to consider when computing the relative change 
            in reward.
        alpha: float
            Decay rate for the exponential moving average.
        threshold: float
            Threshold used for impulse control.
        eps_max: float
            Maximum value of epsilon.
        """
        self.n = n
        self.p = p
        self.alpha = alpha
        self.threshold = threshold
        self.eps_max = eps_max
        self.eps = eps_max
        
    def value(self, t, reward):
        if len(reward) >= self.n:
            m = int(self.p*self.n)
            reward_recent = np.mean(reward[-m:])
            reward_long = np.mean(reward[-self.n:])
            impulse = int(abs((reward_long - reward_recent) / reward_long) < self.threshold)
            self.eps = (self.alpha) * self.eps + (1 - self.alpha) * impulse * self.eps_max     
            return self.eps
        else:
            return self.eps_max * (1 - len(reward) / self.n)