import tensorflow as tf
from uttt.network.architectures import hierarchicalResNet, convNet

model = hierarchicalResNet()

# Text summary (always works, no extra deps)
model.summary(expand_nested=True, show_trainable=True)

# Visual diagram (requires: pip install pydot  +  brew install graphviz)
tf.keras.utils.plot_model(
	model,
	to_file='hierarchical_resnet.png',
	show_shapes=True,
	show_layer_names=True,
	show_layer_activations=True,
	expand_nested=True,
	dpi=150,
)