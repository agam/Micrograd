package main

import (
	"log"
	"os"

	m "github.com/agam/Micrograd"

	"github.com/dominikbraun/graph"
	"github.com/dominikbraun/graph/draw"
)

func traverse(v *m.Value, g graph.Graph[string, string]) {
	g.AddVertex(v.String())
	for _, child := range v.Children {
		g.AddEdge(v.String(), child.String())
		traverse(child, g)
	}
}

func makeTree(v *m.Value) graph.Graph[string, string] {
	g := graph.New(graph.StringHash, graph.Directed(), graph.Acyclic())

	traverse(v, g)

	return g
}

func writeDot(v *m.Value) {
	f, err := os.Create("output.dot")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	_ = draw.DOT(makeTree(v), f)
}

func main() {
	a := m.NewValue(2.0, []*m.Value{})
	b := m.NewValue(3.0, []*m.Value{})
	c := a.Add(b)

	writeDot(c)

}
