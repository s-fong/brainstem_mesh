import os
import numpy
import math
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from opencmiss.zinc.context import Context
from opencmiss.zinc.node import Node
from opencmiss.utils.zinc.field import Field, findOrCreateFieldCoordinates

path = "simcore-rat\\"

vfiles = [v for v in os.listdir(path) if v.endswith('.vtk')]

for vf in vfiles:
    print(vf)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path+vf)
    reader.Update()
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    array = points.GetData()
    numpy_nodes = vtk_to_numpy(array)

    step = 1
    numpy_nodes = [numpy_nodes[i] for i in range(0,len(numpy_nodes),step)]
    outFile = path+vf+'_step%d.exf' %step if step > 1 else path+vf+'.exf'

    context = Context("brainstem")
    region = context.getDefaultRegion()
    fm = region.getFieldmodule()
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    coordinates = findOrCreateFieldCoordinates(fm, "data_coordinates")
    mesh1d = fm.findMeshByDimension(1)
    nodetemplate = nodes.createNodetemplate()
    nodetemplate.defineField(coordinates)
    nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)

    cache = fm.createFieldcache()
    fm.beginChange()
    nodeIdentifier = 1
    for p in numpy_nodes:
        node = nodes.createNode(nodeIdentifier, nodetemplate)
        cache.setNode(node)
        result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, list([float(pp) for pp in p]))
        nodeIdentifier += 1

    fm.endChange()
    region.writeFile(outFile)
