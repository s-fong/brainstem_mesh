# 25 Jan 21 Original file. Each structure in its own region. added simplex triangle elements in 2D.

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from brainstem_python_processing.brainstem_tools import extract_coords_from_opengl, extract_struct_names_opengl
from opencmiss.zinc.context import Context
from opencmiss.zinc.element import Element, Elementbasis
from opencmiss.zinc.field import Field
from opencmiss.zinc.node import Node
from opencmiss.utils.zinc.field import findOrCreateFieldCoordinates, findOrCreateFieldGroup, findOrCreateFieldNodeGroup, findOrCreateFieldStoredString, findOrCreateFieldFiniteElement
import os
from scaffoldmaker.utils.eftfactory_tricubichermite import eftfactory_tricubichermite
import shutil


path = "obj_c_files/"
files = [f for f in os.listdir(path) if f.endswith('.c')]
# files = ['all_structures.c']
files = ['outlines.c']
data = {c:{} for c in files}
dataMerged = {c:[] for c in files}

structNamePath = 'C:\\Users\\sfon036\\Google Drive\\SPARC_work\\brainstem_respiratory_network\\USF_files\\shumanBrainstemSource\\NeuroLabBrainstem\\brainstem\\atlasnames.c'

structNames, namelist = extract_struct_names_opengl(structNamePath)

outline = False
for f in files:
    if 'outlines' in f:
        outline = True

    data, dataMerged = extract_coords_from_opengl(path, f, outline, data, dataMerged, structNames)


    #########################
    # opencmiss - write to ex
    #########################
    skipMultiple = 1
    outFile = path + 'converted_to_ex\\' + f + '.' + str(skipMultiple) + '.exf'
    context = Context("Example")
    region = context.getDefaultRegion()

    shown_structs = []
    for key in data[f].keys():
        if 'norm' not in key.lower():
            shown_structs.append(key)
            if outline:
                fmCh = region.getFieldmodule()
            else:
                childRegion = region.createChild(key)
                fmCh = childRegion.getFieldmodule()
            fmCh.beginChange()
            nodes = fmCh.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            coordinates = findOrCreateFieldCoordinates(fmCh, "data_coordinates")
            norm = findOrCreateFieldFiniteElement(fmCh, "norm", 3)
            nodetemplate = nodes.createNodetemplate()
            nodetemplate.defineField(coordinates)
            nodetemplate.defineField(norm)
            nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
            nodetemplate.setValueNumberOfVersions(norm, -1, Node.VALUE_LABEL_VALUE, 1)
            # mesh1d = fmCh.findMeshByDimension(1)
            mesh2d = fmCh.findMeshByDimension(2)
            # basis1d = fmCh.createElementbasis(1, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
            # eft1d = mesh1d.createElementfieldtemplate(basis1d)
            basisTri = fmCh.createElementbasis(2, Elementbasis.FUNCTION_TYPE_LINEAR_SIMPLEX)
            eftTri = mesh2d.createElementfieldtemplate(basisTri)
            elementtemplate = mesh2d.createElementtemplate()
            elementtemplate.setElementShapeType(Element.SHAPE_TYPE_TRIANGLE)
            result = elementtemplate.defineField(coordinates, -1, eftTri)
            cache = fmCh.createFieldcache()
            # create nodes
            if outline:
                exteriorGroup = findOrCreateFieldGroup(fmCh, 'brainstem_exterior_obsolete')
                exteriorPoints = findOrCreateFieldNodeGroup(exteriorGroup, nodes).getNodesetGroup()
                medullaGroup = findOrCreateFieldGroup(fmCh, 'medulla oblongata_exterior')
                medullaPoints = findOrCreateFieldNodeGroup(medullaGroup, nodes).getNodesetGroup()
                ponsGroup = findOrCreateFieldGroup(fmCh, 'pons_exterior')
                ponsPoints = findOrCreateFieldNodeGroup(ponsGroup, nodes).getNodesetGroup()
                midbrainGroup = findOrCreateFieldGroup(fmCh, 'midbrain_exterior')
                midbrainPoints = findOrCreateFieldNodeGroup(midbrainGroup, nodes).getNodesetGroup()
                extNodetemplate = exteriorPoints.createNodetemplate()
                extNodetemplate.defineField(coordinates)
                extNodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
                nEnd = len(dataMerged[f])
                xs = dataMerged[f]
                # marker/group node numbers found manually in cmgui
                markerNodeIdentifiers = [3, 70, 1419,1515, 4161, 4292]
                medullaRange = [1, 1613]
                ponsRange = [1614, 3932]
                midbrainRange = [3933, 4424]
                markerNames = ['caudal-dorsal', 'caudal-ventral', 'midRostCaud-dorsal', 'midRostCaud-ventral', 'rostral-dorsal', 'rostral-ventral']
            else:
                nEnd = len(data[f][key]['xyz'])
                xs = data[f][key]['xyz']
            markerNodeIdentifier = 1
            markerx = []
            for nodeIdentifier in range(1, nEnd + 1):
                x = xs[nodeIdentifier - 1]
                if outline and not nodeIdentifier % skipMultiple: # filter the number of nodes going into brainstem_exterior group
                    node = exteriorPoints.createNode(nodeIdentifier, extNodetemplate)
                    if nodeIdentifier <= medullaRange[1] and nodeIdentifier >= medullaRange[0]:
                        medullaPoints.addNode(node)
                    elif nodeIdentifier <= ponsRange[1] and nodeIdentifier >= ponsRange[0]:
                        ponsPoints.addNode(node)
                    else:
                        midbrainPoints.addNode(node)
                else:
                    nv = data[f][key]['norm'][nodeIdentifier - 1]
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                cache.setNode(node)
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                if outline and nodeIdentifier in markerNodeIdentifiers:
                    markerx.append(x)
                if not outline:
                    norm.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, nv)

            # create elements
            if not outline:
                for elementIdentifier in range(1, int(nEnd/3)+1):
                    nodeIdentifiers = [(elementIdentifier*3)-2, (elementIdentifier*3)-1, elementIdentifier*3]
                    element = mesh2d.createElement(elementIdentifier, elementtemplate)
                    result = element.setNodesByIdentifier(eftTri, nodeIdentifiers)
                    elementIdentifier += 1
                fmCh.defineAllFaces()
                fmCh.endChange()

            else:
                # fmCh = region.getFieldmodule()
                # fmCh.beginChange()
                nodes = fmCh.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
                markerGroup = findOrCreateFieldGroup(fmCh, 'marker')
                markerPoints = findOrCreateFieldNodeGroup(markerGroup, nodes).getNodesetGroup()
                markerName = findOrCreateFieldStoredString(fmCh, name='marker_data_name')
                markerdatacoordinates = findOrCreateFieldCoordinates(fmCh, "marker_data_coordinates")
                markertemplate = markerPoints.createNodetemplate()
                markertemplate.defineField(markerdatacoordinates)
                markertemplate.setValueNumberOfVersions(markerdatacoordinates, -1, Node.VALUE_LABEL_VALUE, 1)
                markertemplate.defineField(markerName)
                cache = fmCh.createFieldcache()

                for im in range(len(markerNodeIdentifiers)):
                    markerNodeIdentifier = 100000 + im + 1
                    node = markerPoints.createNode(markerNodeIdentifier, markertemplate)
                    cache.setNode(node)
                    res = markerdatacoordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, markerx[im])
                    res = markerName.assignString(cache, markerNames[im])
    if outline:
        fmCh.endChange()
        region.getFieldmodule().defineAllFaces()
    else:
        # display the largest and smallest norms
        normSmall = 1000
        normLarge = 1
        for key in data[f].keys():
            normMags = [np.linalg.norm(v) for v in data[f][key]['norm']]
            smallCurrent, largeCurrent = [min(normMags), max(normMags)]
            normSmall = smallCurrent if smallCurrent < normSmall else normSmall
            normLarge = largeCurrent if largeCurrent > normLarge else normLarge
        print('smallest norm is ' + str(normSmall))
        print('largest norm is ' + str(normLarge))

    region.writeFile(outFile)
    # copy file into data folder of mapclient workflow
    workflow_path = "C:\\Users\\sfon036\\Google Drive\\SPARC_work\\codes\\mapclient_workflows\\brainstem_nuclear_groups_only\\data\\"
    shutil.copyfile(outFile, workflow_path+outFile.split('\\')[-1])

    # write com file to view with all subregions
    cols = ["gold","silver","green","cyan","magenta","orange","yellow","red","white"]
    if files == ['all_structures.c']:

        outputComFile = path + 'view_converted_ex_all_structures.c.com'
        with open(outputComFile,'w') as w_out:
            w_out.write('gfx read nodes "converted_to_ex\\all_structures.c.%d.exf"\n\n' %(skipMultiple))
            w_out.write("gfx modify g_element "" / "" general clear;\n")
            for c,key in enumerate(shown_structs):

                if 'brain' in key:
                    w_out.write("gfx modify g_element /%s/ general clear;\n" % (key))
                    w_out.write("gfx modify g_element /%s/ points domain_nodes coordinate data_coordinates tessellation default_points LOCAL glyph sphere size ""0.1*0.1*0.1"" offset 0,0,0 font default select_on material orange selected_material default_selected render_shaded;\n" %(key))
                    w_out.write("gfx modify g_element \"/%s/\" surfaces domain_mesh2d coordinate data_coordinates face all tessellation default LOCAL select_on invisible material yellow selected_material default_selected render_shaded;\n" %(key))
                else:
                    w_out.write("gfx modify g_element \"/%s/\" general clear;\n" % (key))
                    w_out.write("gfx modify g_element \"/%s/\" points domain_nodes coordinate data_coordinates tessellation default_points LOCAL glyph arrow_solid size ""0.1*0.1*0.1"" offset 0,0,0 font default orientation norm scale_factors ""0.1*0*0"" select_on invisible material default selected_material default_selected render_shaded;\n" %(key))
                    w_out.write("gfx modify g_element \"/%s/\" surfaces domain_mesh2d coordinate data_coordinates face all tessellation default LOCAL select_on material %s selected_material default_selected render_shaded;\n" %(key,cols[c%9]))
                w_out.write("gfx modify g_element \"/%s/\" lines domain_mesh1d coordinate data_coordinates face all tessellation default LOCAL line line_base_size 0 select_on invisible material default selected_material default_selected render_shaded;\n" %(key))


            w_out.write("\ngfx create window\n")
            w_out.write("gfx edit scene\n")
