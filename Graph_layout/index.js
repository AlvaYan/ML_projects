var dagre = require("dagre");
var saveAs = require("file-saver").saveAs;
fs = require('fs');

// Create a new directed graph 
var g = new dagre.graphlib.Graph();

// Set an object for the graph label
g.setGraph({ compound: true });
g.setGraph({ rankdir: "LR", node: { shape: "rectangle", style: "rounded" } });
  g.setDefaultEdgeLabel(function () {
    return {};
  });

//load data
var dag= require('./data.json');

//add nodes
let length1 = dag['n_nodes'];
for (let i = 0; i< length1; i++) {
    g.setNode(dag['nodes'][i]['label'], {width: dag['nodes'][i]['width'], height: dag['nodes'][i]['height']})
}

//add edges
let length2 = dag['n_edges'];
for (let i = 0; i< length2; i++) {
    g.setEdge(dag['edges'][i]['parent'],  dag['edges'][i]['child']);
}

const t1 = performance.now();
dagre.layout(g, { rankdir: "LR" });
const t2 = performance.now();

//save
/*
let name1 = "time" + ".txt";
var gr = new Blob([JSON.stringify(t2-t1)], {
    type: "text/plain;charset=utf-8"
  });
saveAs(gr, name1)

let name2 = "graph" + ".txt";
var gr = new Blob([JSON.stringify(g)], {
    type: "text/plain;charset=utf-8"
  });
saveAs(gr, name2)
*/

fs.writeFile('graph.txt', JSON.stringify(g), encoding='utf8',callback=function(err) {
    if(err) {
        return console.log(err);
    }
    console.log("The file was saved!");
});
fs.writeFile('time.txt', JSON.stringify(t2-t1), encoding='utf8',callback=function(err) {
    if(err) {
        return console.log(err);
    }
    console.log("The file was saved!");
});