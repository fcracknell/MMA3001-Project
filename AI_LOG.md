used to create a suggested initial framework/project workflow based on my proposed concept

Claude designed the original split with validation in the same period as training. Run 1 showed the corrections made predictions worse on unseen objects; the cause was traced to this and fixed in run 2.

Claude used cProfile output to find the history-feature bottleneck and vectorised it; I verified it with a test confirming identical results, and measured a 5.3x speed-up.

