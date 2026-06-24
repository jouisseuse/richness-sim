Subject: Quick update on the richness simulations

Hi Youngjae,

Sorry for the late reply. I hope you’re having a great summer. I took a few days off recently and am just getting back to work.

I spent some time exploring the two ideas we discussed and wanted to share a quick update.

The first idea was to formalize the emergence of culture using a gossip-based bandit framework. That direction did not work as well as I had hoped. In the current setup, agents did not converge on just one or two arms, so we did not observe the kind of cultural lock-in I was expecting. My sense is that simple information-sharing is not enough to generate culture-like convergence; we may need a stronger imitation, prestige, or majority-signal mechanism for that model to work.

The second idea was to formalize ecological richness using a simple bandit framework. Inspired by your Gangnam/Gangbuk example, the core question was:

**Does ecological richness automatically become experienced richness?**

The short answer from the simulations is no.

The model ended up separating richness into three layers:

1. **Ecological richness**: all possible opportunities in the environment.
2. **Feasible richness**: opportunities that are actually accessible given constraints such as time and cost.
3. **Experienced richness**: opportunities that are actually explored.

One interesting result is that environments with many possible opportunities do not necessarily produce proportionally richer lived experiences. For example, a “big city” condition contained many more possible options, but many of them remained unexplored because they were not feasible. In contrast, a smaller ecology often had a much higher conversion rate from possible experiences to lived experiences.

Another distinction that emerged from the simulations is between:

1. **Stock richness**: the accumulated diversity of experiences.
2. **Flow richness**: the ongoing rate of novelty over time.

This ended up being the most interesting result. Rich ecologies increase stock richness, but novelty tends to decay as agents settle into routines. In contrast, environments that continuously generate new feasible opportunities maintain higher flow richness, even when total accumulated experiences are lower.

I also explored a breadth-versus-depth extension. In that version, richness can come either from sampling many different experiences or from repeated engagement with experiences that continue to deepen over time. This suggests that psychologically rich lives may not always require a large number of options; richness could also come from depth within a smaller ecology.

My current takeaway is that psychological richness may emerge not simply from the number of opportunities available in an ecology, but from the conversion of ecological possibilities into lived novelty. More broadly, the simulations suggest that richness may involve both:

- breadth versus depth
- stock versus flow

I’d love to hear what you think and whether these results connect to how you conceptualize psychological richness. If you’re interested, I’d be happy to meet sometime and walk through the simulations and figures.

Best,
Bufan
