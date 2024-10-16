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

func NewConst(data float64) *Value {
	value := &Value{Data: data, Children: []*Value{}, Op: "", Label: "CONST", Grad: 0.0}
	value.Backward = func() {}
	return value
}

func NewValue(data float64, label string) *Value {
	value := &Value{Data: data, Children: []*Value{}, Op: "", Label: label, Grad: 0.0}
	value.Backward = func() {}
	return value
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

func (v *Value) Sub(other *Value, label string) *Value {
	return v.Add(other.Neg(" -neg "), label)
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

func (v *Value) Tanh(label string) *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := NewValueWithOp(t, []*Value{v}, " tanh ", label)
	backward := func() {
		v.Grad += (1 - t*t) * out.Grad
	}
	out.Backward = backward

	return out
}

func (v *Value) Exp(label string) *Value {
	out := NewValueWithOp(math.Exp(v.Data), []*Value{v}, " exp ", label)
	backward := func() {
		v.Grad += out.Grad * out.Data
	}
	out.Backward = backward
	return out
}

func (v *Value) Pow(other *Value, label string) *Value {
	out := NewValueWithOp(math.Pow(v.Data, other.Data), []*Value{v, other}, " pow ", label)
	backward := func() {
		v.Grad += other.Data * math.Pow(v.Data, other.Data-1) * out.Grad
	}
	out.Backward = backward
	return out
}

func (v *Value) Div(other *Value, label string) *Value {
	return v.Pow(other.Neg(label), label)
}

func (v *Value) Neg(label string) *Value {
	return v.Mul(NewConst(-1.0), label)
}

func (v *Value) String() string {
	if v.Op == "" {
		return fmt.Sprintf("Value(%s => %f) [ GRAD: %f ]", v.Label, v.Data, v.Grad)
	}
	return fmt.Sprintf("OP [ %s ] %s => %f [ GRAD: %f ]", v.Op, v.Label, v.Data, v.Grad)
}

func getTopologicalOrder(v *Value) []*Value {
	order := []*Value{}
	visited := map[*Value]bool{}

	var dfs func(v *Value)
	dfs = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, child := range v.Children {
			dfs(child)
		}
		order = append(order, v)
	}
	dfs(v)

	// Reverse the order to get the topological order
	for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
		order[i], order[j] = order[j], order[i]
	}

	return order
}

func (v *Value) BackProp() {
	order := getTopologicalOrder(v)
	for _, node := range order {
		node.Backward()
	}
}

func Loss(ys, ypred []*Value) *Value {
	out := NewValue(0, "loss")
	for i, y := range ys {
		diff := y.Sub(ypred[i], fmt.Sprintf("(%f - %f)", y.Data, ypred[i].Data))
		diffSquared := diff.Pow(NewConst(2), fmt.Sprintf("(%f - %f)^2", y.Data, ypred[i].Data))
		out = out.Add(diffSquared, fmt.Sprintf(" Loss - %d ", i))
	}
	return out
}
