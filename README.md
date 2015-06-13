# Sia-GPU-Miner
Miner with some OpenCL initialization fixed, and several options to allow the user decent control.

## How to Use
Usage:

 I - intensity: This is the amount of work sent to the GPU in one batch.
	Interpretation is 2^intensity; the default is 16. Lower if GPU crashes or
	if more desktop interactivity is desired. Raising it may improve performance.

 p - OpenCL platform ID: Just what it says on the tin. If you're finding no GPUs,
	yet you're sure they exist, try a value other than 0, like 1, or 2. Default is 0.

 d - OpenCL device ID: Self-explanatory; it's the GPU index. Note that different
	OpenCL platforms will likely have different devices available. Default is 0.

 C - cycles per iter: Number of kernel executions between Sia API calls and hash rate updates
	Increase this if your miner is receiving invalid targets. Default is 3.

## Notes
*    Each Sia block takes about 10 minutes to mine.
*    Once a block is mined, Sia waits for 144 confirmation blocks before the reward is added to your wallet, which takes about 24 hours.
*    Sia currently doesn't have any mining pools. A p2pool portal is under development, and can be expected to be ready by the end of the summer.
