# GPT-CPP architecture and training analysis

## Architecture match with GPT

This project **does implement a decoder-only Transformer language model (GPT-style)** at a high level:

- Token + positional embeddings are created and summed before the stack of blocks.
- Each block applies causal self-attention + feed-forward network with residual connections and pre-layer normalization.
- Final normalized hidden states are projected to vocabulary logits using tied embeddings, then softmax/cross-entropy is used for next-token prediction.

## Confirmed strengths

- Causal masking is applied before softmax in attention.
- LM head is tied to the embedding matrix in forward and backward.
- Manual backprop is implemented for FFN and attention paths.
- Training loss decreases on the provided sample data when run from `src/`.

## Key problems that can break or degrade training

1. **Path handling is brittle**
   - Many paths use relative `../...` assumptions (data/model/config).
   - Running the program from repo root fails to find files and silently prints errors.

2. **Configuration typo can cause runtime mismatch**
   - Key name `Emdedding_size` is misspelled in both code/config. It works only because both sides share the same typo.

3. **No train/validation split or evaluation loop**
   - Only a single contiguous sequence is trained repeatedly.
   - No held-out perplexity/accuracy, so overfitting and quality are not measurable.

4. **Single full-context batch only**
   - `X` and `Y` are built from one shifted sequence; there is no batching, shuffling, or random window sampling.
   - This limits data efficiency and gradient quality.

5. **Optimization is very minimal**
   - Plain SGD-style updates with fixed learning rate.
   - No AdamW, weight decay decoupling, learning-rate warmup, gradient clipping, or scheduler.

6. **LayerNorm is incomplete compared to GPT**
   - Normalization has no learned gain/bias parameters.
   - That can reduce model expressivity/stability relative to standard GPT blocks.

7. **Potential scaling/robustness risks**
   - No mixed precision, checkpointing, or memory-efficient batching.
   - Code relies heavily on dynamic nested vectors; this can become slow and memory-fragmented for larger models.

8. **Tokenizer/training consistency risks**
   - BPE/vocab files are mutable and can be rewritten depending on flow.
   - If vocabulary grows but `Vocab_size` in config is stale, embedding index errors may occur.

## Practical recommendations

- Normalize path management (absolute project root resolution, CLI args, or config-driven paths).
- Add dataset pipeline with random context windows + mini-batching.
- Add validation split and report perplexity over time.
- Replace optimizer with AdamW + LR warmup/decay + grad clipping.
- Add gamma/beta parameters for layer norm.
- Make vocab size derived from tokenizer artifacts at runtime, with explicit mismatch checks.
- Add model checkpoint save/load and reproducibility controls (seeds).

## Bottom line

- **Is it GPT architecture?** Yes, it is a simplified GPT-like decoder-only Transformer.
- **Any training problems?** Yes: mainly data/optimization/evaluation pipeline limitations, brittle file paths, and a few implementation choices that reduce robustness compared with production-grade GPT training.
