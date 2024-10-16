package main

import (
	"fmt"

	m "github.com/agam/Micrograd"
)

func simpleNN() {
	x1 := m.NewValue(2.0, "x1")
	x2 := m.NewValue(0.0, "x2")

	w1 := m.NewValue(-3.0, "w1")
	w2 := m.NewValue(1.0, "w2")

	b := m.NewValue(6.881373587629085, "b")

	x1w1 := x1.Mul(w1, "x1*w1")
	x2w2 := x2.Mul(w2, "x2*w2")

	x1w1x2w2 := x1w1.Add(x2w2, "x1*w1 + x2*w2")

	nn := x1w1x2w2.Add(b, "nn")
	o := nn.Tanh("o")

	o.Grad = 1.0
	o.BackProp()

	WriteDot(o)
}

func SimpleNeuron() {
	n := m.NewNeuron(2)
	x := []*m.Value{m.NewValue(2.0, "x1"), m.NewValue(0.0, "x2")}
	o := n.Call(x)
	WriteDot(o)
}

func SimpleLayer() {
	l := m.NewLayer(2, 3)
	x := []*m.Value{m.NewValue(2.0, "x1"), m.NewValue(3.0, "x2")}
	o := l.Call(x)
	for _, v := range o {
		WriteDot(v)
	}
}

func SimpleMLP() {
	mlp := m.NewMLP(3, []int{4, 4, 1})
	x := []*m.Value{m.NewValue(2.0, "x1"), m.NewValue(3.0, "x2"), m.NewValue(-1.0, "x3")}
	o := mlp.Call(x)
	WriteDot(o[0])
}

func RealMLP() {
	xs := [][]*m.Value{
		{m.NewValue(2.0, "x11"), m.NewValue(3.0, "x12"), m.NewValue(-1.0, "x13")},
		{m.NewValue(3.0, "x21"), m.NewValue(-1.0, "x22"), m.NewValue(0.5, "x23")},
		{m.NewValue(0.5, "x31"), m.NewValue(1.0, "x32"), m.NewValue(1.0, "x33")},
		{m.NewValue(1.0, "x41"), m.NewValue(1.0, "x42"), m.NewValue(-1.0, "x43")},
	}
	ys := []*m.Value{
		m.NewValue(1.0, "y1"),
		m.NewValue(-1.0, "y2"),
		m.NewValue(-1.0, "y3"),
		m.NewValue(1.0, "y4"),
	}
	mlp := m.NewMLP(3, []int{4, 4, 1})
	ypred := make([]*m.Value, len(xs))
	for i, x := range xs {
		ypred[i] = mlp.Call(x)[0]
	}
	loss := m.Loss(ys, ypred)
	loss.Grad = 1.0
	loss.BackProp()

	fmt.Printf("ypred: %v\n", ypred)
	fmt.Printf("Loss => %f\n", loss.Data)
	for i, n := range mlp.Layers[0].Neurons {
		fmt.Printf("[L0] Neuron %d => %f [W0]\n", i, n.Weights[0].Grad)
	}

	WriteDot(loss)
}

func main() {
	RealMLP()
}
