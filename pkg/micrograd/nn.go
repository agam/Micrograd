package micrograd

import (
	"fmt"
	"math"
)

type Value struct {
	Data     float64
	Children []*Value
	Op       string
	Label    string
	Grad     float64
	Backward func()
}

func NewValue(data float64, label string) *Value {
	return &Value{Data: data, Children: []*Value{}, Op: "", Label: label, Grad: 0.0}
}

func NewValueWithOp(data float64, children []*Value, op string, label string) *Value {
	return &Value{Data: data, Children: children, Op: op, Label: label}
}

func (v *Value) Add(other *Value, label string) *Value {
	out := NewValueWithOp(v.Data+other.Data, []*Value{v, other}, " + ", label)
	backward := func() {
		v.Grad += 1.0 * out.Grad
		other.Grad += 1.0 * out.Grad
	}
	out.Backward = backward
	return out
}

func (v *Value) Mul(other *Value, label string) *Value {
	out := NewValueWithOp(v.Data*other.Data, []*Value{v, other}, " * ", label)
	backward := func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	out.Backward = backward
	return out
}

func (v *Value) Tanh() *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := NewValueWithOp(t, []*Value{v}, " tanh", v.Label)
	backward := func() {
		v.Grad += (1 - t*t) * out.Grad
	}
	out.Backward = backward
	out.Children = []*Value{v}

	return out
}

func (v *Value) String() string {
	if v.Op == "" {
		return fmt.Sprintf("Value(%s => %f)", v.Label, v.Data)
	}
	return fmt.Sprintf("OP [ %s ] %s => %f", v.Op, v.Label, v.Data)
}
