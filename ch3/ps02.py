import numpy as np
import pylab as plt

if __name__ == "__main__":

    alpha_prior = np.array([1., 1., 1.]) # uninformative
    data_before_debate = np.array([294, 307, 38]) # see Table 3.2
    data_after_debate = np.array([288, 332, 19])

    K = 8192 # MAGIC
    samples_before = np.random.dirichlet(alpha_prior + data_before_debate, size=K)
    samples_after = np.random.dirichlet(alpha_prior + data_after_debate, size=K)

    # "alpha" is the preference for the alpha male
    alphas_before = samples_before[:,0] / (samples_before[:,1] + samples_before[:,0])
    alphas_after = samples_after[:,0] / (samples_after[:,1] + samples_after[:,0])

    # scatter plot
    plt.clf()
    plt.plot(alphas_before, alphas_after, "k.", alpha=0.25)
    plt.plot([0., 1.], [0., 1.], "k-", alpha=0.25)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("alpha before")
    plt.ylabel("alpha after")
    plt.savefig("ps02a.png")

    # histogram
    plt.clf()
    plt.hist(alphas_after - alphas_before, histtype="step", bins=200)
    plt.axvline(0., color="k", alpha=0.25)
    plt.xlabel("alpha after minus alpha before")
    plt.savefig("ps02b.png")

    # answer the question, dammit!
    p_bad = np.sum(alphas_after > alphas_before) / K
    print("The approximate probability of alpha increase is", p_bad)
