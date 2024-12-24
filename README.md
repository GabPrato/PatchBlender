# PatchBlender
Implementation of the PatchBlender layer introduced in https://arxiv.org/abs/2211.14449

## Instructions
This torch module can be used in various contexts; in the paper, we use it between the layer normalization and the attention layer.
```python
# Transformer layer example
def forward(self, x):
    residual = x
    x = self.layer_norm1(x)
    x = self.patch_blender(x)
    x = self.attention_layer(x)
    x += residual

    residual = x
    x = self.layer_norm2(x)
    x = self.feed_forward(x)
    x += residual

    return x
```
