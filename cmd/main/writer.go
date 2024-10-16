package main

import (
	"log"
	"os"

	m "github.com/agam/Micrograd/pkg/micrograd"

	"github.com/dominikbraun/graph"
	"github.com/dominikbraun/graph/draw"
)

func addVertices(v *m.Value, g graph.Graph[string, string]) {
	g.AddVertex(v.String())
	for _, child := range v.Children {
		addVertices(child, g)
	}
}

func addEdges(v *m.Value, g graph.Graph[string, string]) {
	for _, child := range v.Children {
		g.AddEdge(child.String(), v.String())
		addEdges(child, g)
	}
}

func makeTree(v *m.Value) graph.Graph[string, string] {
	g := graph.New(graph.StringHash, graph.Directed(), graph.Acyclic())

	addVertices(v, g)
	addEdges(v, g)

	return g
}

func WriteDot(v *m.Value) {
	f, err := os.Create("output.dot")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	_ = draw.DOT(makeTree(v), f)
}
