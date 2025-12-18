# Tasks for Data Generation
The `tasks` directory contains (1) an implementation of the generator for our main in-context algebra task, as well as (2) an implementation of various non-group algebraic structures. 

> (1) `mix_group_addition.py` contains the main code we use to generate sequences for our in-context algebra task. Specifically, the `MixGroupAddition` base class generates sequences that simulate a mixture of finite groups, with classes extending it for different group mixtures.

**These utility methods are used for generation of algebra sequences:**
* `sample_run` - generates a single sequence (as a string)
* `sample_batch` - generates a batch of sequences (as a torch tensor)

**These utility methods handle "tokenization" of sequences:**
* `stringify` - converts a tensor sequence to its string version
* `tensor_from_expression` - converts a string to its tensor version

The following code is an example of how to generate a random algebra sequence (from the training distribution).

```
from src.tasks.mix_group_addition import MixRosetteGroupAddition
task = MixRosetteGroupAddition(num_symbols=16, max_order=10, mix=0.7)

sequence, groups, orders, vocab = task.sample_run(k_shots=200, hold_out=False,commute_out=False, unshuffled=False, fixed_groups=None)

inputs = task.tensor_from_expression([sequence])
```

> (2) `magma.py` contains our implementations of various non-group algebraic structures such as magmas, semigroups, and quasigroups. They are constructed in a similar manner to `PermutationGroup` classes found in sympy.combinatorics  such as `CyclicGroup` and `DihedralGroup`


___ 

Other files that contain data-generation code for specific experiments can be found in `src/data_utils`.