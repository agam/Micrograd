package main

import (
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

func main() {
	SimpleLayer()
}
