package micrograd

import (
	"fmt"
	"math/rand/v2"
)

type Neuron struct {
	Weights []*Value
	Bias    *Value
}

func NewNeuron(numInputs int) *Neuron {
	weights := make([]*Value, numInputs)
	for i := 0; i < numInputs; i++ {
		val := rand.Float64()*2.0 - 1.0
		weights[i] = NewValue(val, fmt.Sprintf("weight_%d", i))
	}
	bias := NewValue(rand.Float64()*2.0-1.0, "bias")
	return &Neuron{Weights: weights, Bias: bias}
}

func (n *Neuron) Call(inputs []*Value) *Value {
	sum := NewConst(0.0)
	for i, input := range inputs {
		sum = sum.Add(input.Mul(n.Weights[i], fmt.Sprintf("sum_%d", i)), fmt.Sprintf("sum_%d", i))
	}
	sum = sum.Add(n.Bias, "activation-sum+bias")
	out := sum.Tanh(" activation-out ")
	return out
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(numInputs, numNeurons int) *Layer {
	neurons := make([]*Neuron, numNeurons)
	for i := 0; i < numNeurons; i++ {
		neurons[i] = NewNeuron(numInputs)
	}
	return &Layer{Neurons: neurons}
}

func (l *Layer) Call(inputs []*Value) []*Value {
	outputs := make([]*Value, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Call(inputs)
	}
	return outputs
}

type MLP struct {
	Layers []*Layer
}

func NewMLP(numInputs int, numOutputs []int) *MLP {
	numNeurons := append([]int{numInputs}, numOutputs...)
	layers := make([]*Layer, len(numNeurons)-1)
	for i := 0; i < len(numNeurons)-1; i++ {
		layers[i] = NewLayer(numNeurons[i], numNeurons[i+1])
	}
	return &MLP{Layers: layers}
}

func (m *MLP) Call(inputs []*Value) []*Value {
	for _, layer := range m.Layers {
		inputs = layer.Call(inputs)
	}
	return inputs
}
