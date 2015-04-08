import contextlib

import numpy as np
import pandas as pd
import pylab
try:
    # TODO maybe remove?
    # optionally use seaborn if available
    import seaborn
except ImportError:
    pass


@contextlib.contextmanager
def plot_to(filename=None):
    yield
    if filename is None:
        pylab.show()
    else:
        pylab.savefig(filename)
    pylab.clf()


def plot_objective_values(states, labels, title=None):
    """
    plot objective values over time for multiple states
    """
    upper_bound = max([state.get("objective_upper_bound", -np.inf)
                       for state in states])
    series = [pd.Series(state["objective_values"], index=state["num_labeled"])
              for state in states]
    df = pd.DataFrame.from_items([(label, s)
                                  for label, s in zip(labels, series)])
    df.plot(title=title)
    pylab.xlabel("labeled data points")
    pylab.ylabel("objective value")
    if not np.isinf(upper_bound):
        pylab.axhline(upper_bound)
    return df


def _plot_labeled_unlabeled_score_distibution(labeled, unlabeled):
    """
    rationale: scores should be a proxy for how much information a new sample
    will give, thus the distribution of scores for the labeled examples
    should be lower than that of the unlabeled examples (else the algorithm
    will choose points similar to those already labeled, which might be
    suboptimal)
    """
    pylab.hist(labeled,
               alpha=0.5,
               color="red",
               normed=True,
               label=["labeled"])
    pylab.hist(unlabeled,
               alpha=0.5,
               color="blue",
               normed=True,
               label=["unlabeled"])
    pylab.legend()


def plot_labeled_unlabeled_score_distibutions(state):
    labeled = state["labeled_scores"]
    unlabeled = state["unlabeled_scores"]
    pylab.subplot(211)
    _plot_labeled_unlabeled_score_distibution(labeled[0],
                                              unlabeled[0])
    pylab.title("Initial/final distribution")
    pylab.subplot(212)
    _plot_labeled_unlabeled_score_distibution(labeled[-1],
                                              unlabeled[-1])


def animate_labeled_unlabeled_score_distibutions(state, filename=None):
    obj = state["objective_values"]
    labeled = state["labeled_scores"]
    unlabeled = state["unlabeled_scores"]
    all_mins = min(ys.min()
                   for xs in [labeled, unlabeled]
                   for ys in xs)
    all_maxs = max(ys.max()
                   for xs in [labeled, unlabeled]
                   for ys in xs)

    def animate(nframe):
        pylab.clf()
        pylab.subplot(211)
        pylab.xlim(all_mins, all_maxs)
        _plot_labeled_unlabeled_score_distibution(labeled[nframe],
                                                  unlabeled[nframe])
        pylab.title('Label %d' % nframe)
        pylab.subplot(212)
        pylab.plot(obj[:nframe + 1])

    fig = pylab.figure()
    from matplotlib import animation
    anim = animation.FuncAnimation(fig,
                                   animate,
                                   frames=len(labeled),
                                   # 100ms per frame
                                   interval=100,
                                   repeat=True,
                                   blit=True,
                                   # 1 second (doesn't seem to work)
                                   repeat_delay=1000)
    if filename is not None:
        anim.save(filename, writer='imagemagick')
    else:
        pylab.show()
    pylab.clf()
