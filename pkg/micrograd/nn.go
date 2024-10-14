package micrograd

import "fmt"

type Value struct {
	Data     float64
	Children []*Value
	Op       string
}

func NewValue(data float64, children []*Value) *Value {
	return &Value{Data: data, Children: children}
}

func (v *Value) Add(other *Value) *Value {
	return NewValue(v.Data+other.Data, []*Value{v, other})
}

func (v *Value) Mul(other *Value) *Value {
	return NewValue(v.Data*other.Data, []*Value{v, other})
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(%f)", v.Data)
}
