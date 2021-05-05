from scipy.spatial.distance import pdist, squareform
from numpy import nanmax, argmax, unravel_index
from math import *
from numpy import linalg as LA
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from opencmiss.zinc.context import Context
from opencmiss.zinc.element import Element, Elementbasis
from opencmiss.zinc.node import Node
from opencmiss.utils.zinc.field import Field, findOrCreateFieldFiniteElement, findOrCreateFieldCoordinates, findOrCreateFieldGroup, \
    findOrCreateFieldNodeGroup, findOrCreateFieldStoredMeshLocation, findOrCreateFieldStoredString
from opencmiss.utils.zinc.finiteelement import getElementNodeIdentifiersBasisOrder
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup
from scaffoldmaker.utils.meshrefinement import MeshRefinement


def cranial_nerve_names():
    cranialDict = {
        1: 'OLFACTORY',
        2: 'OPTIC',
        3: 'OCULOMOTOR',
        4: 'TROCHLEAR',
        5: 'TRIGEMINAL',
        6: 'ABDUCENS',
        7: 'FACIAL',
        8:'VESTIBULOCOCHLEAR',
        9:'GLOSSOPHARYNGEAL',
        10: 'VAGUS',
        11: 'ACCESSORY',
        12: 'HYPOGLOSSAL'
    }
    return cranialDict


def extract_coords_from_opengl(path, f, outline, data, structNames, wantNorm = 0):
    # if not dataMerged:
    #     dataMerged = data.copy()
    tempD = {}
    # tempDM = []

    use_set = True
    use_set = False if outline else use_set

    with open(path+f,'r') as f_in:
        print('Reading ' + f)
        found = False
        found_norm = False
        xyz = []
        norm = []
        linecount = 0
        for line in f_in:
            linecount += 1
            if 'GLfloat' in line and '*' not in line and 'int' not in line:
                found = True
                if outline:
                    # objType = line.split('GLfloat ')[-1].split('[')[0]
                    objType = "brainstem_exterior"
                else:
                    objType = line.split('GLfloat ')[-1].split('[]')[0]
                    if 'brain' not in objType and 'skin' not in objType and 'norm' not in objType.lower():
                        objID = int(objType.split('object')[-1])
                        objType = structNames[objID]
                if 'Norms[]' not in line or 'Norm[]' not in line:
                    line = f_in.readline()

            if found:
                if ';' in line:
                    if xyz and norm and not outline:
                        # remove duplicate values from xyz and the corresponding norm values
                        xyzUnique = []
                        normUnique = []
                        if use_set:
                            # xyzUnique = list(set(xyz))
                            xyzUnique = [list(x) for x in set(tuple(t) for t in xyz)]
                        else:
                            for count, ix in enumerate(xyz):
                                if ix not in xyzUnique:
                                    xyzUnique.append(ix)
                                    normUnique.append(norm[count])

                        tempD.update({objType: {'xyz': xyzUnique, 'norm': normUnique}})
                        # tempDM.extend(xyzUnique)
                        xyz = []
                        norm = []
                        found = False
                        found_norm = False

                    elif xyz and outline:
                        # remove duplicate values from xyz and the corresponding norm values
                        xyzUnique = []
                        if use_set:
                            xyzUnique = [list(x) for x in set(tuple(t) for t in xyz)]
                        else:
                            for ix in xyz:
                                if ix not in xyzUnique:
                                    xyzUnique.append(ix)

                        tempD.update({objType: {'xyz': xyzUnique}})
                        # tempDM.extend(xyzUnique)
                        xyz = []
                        found = False

                elif line == '};\n' or line == '\n':
                    line = f_in.readline()
                    if 'Norms[]' in line or 'Norm[]' in line:
                        found_norm = True
                elif found_norm:
                    if wantNorm:
                        norm.append([float(a) for a in line.replace('{', '').replace('},', '').split('\n')[0].split(',')])
                    else:
                        norm.append([])
                elif '&' in line or '0,\n' in line:
                    break
                else:
                    # xyz.append([float(a) for a in line.replace('{', '').replace('},', '').split('\n')[0].split(',')])
                    xyz.append([float(a) for a in line.split(',')])

    if data.keys():
        data[f] = tempD.copy()
        # dataMerged[f] = tempDM.copy()
    else:
        data = tempD.copy()
        # dataMerged = tempDM.copy()
    return data


def extract_struct_names_opengl(path):
    structNames = {}
    namelist = []

    # get atlas names. can't port structure in using ctypes as indices are in the COMMENTS ?!?!
    with open(path, 'r') as sn:
        start = False
        for line in sn:
            if '*structNames' in line:
                start = True
                line = sn.readline()
                line = sn.readline()
            elif ';' in line:
                start = False
            if start:
                linesp = line.replace('"', '').split(',//')
                name = linesp[0][1:]  # don't include leading space character
                index = int(linesp[-1].split('=')[-1])
                structNames.update({index: name})
                namelist.append(name)
    return structNames, namelist


def centroids_of_tract(points):
    # assume that datapoints are made of discrete slices which may not be parallel to xyz axes. On each slice, find the centroid of points. Retain order of centroids to connect with 1D elements.
    line = []
    ztol = 0.3 # manually chose this
    z_vals = sorted(set(points[:,2]))
    zbins = [10000]
    for z in z_vals:
        if z not in zbins and abs(z-zbins[-1]) > ztol:
            zbins.append(z)
    del zbins[0]

    for zbin in zbins:
        slice = [row for row in points if abs(row[2]-zbin)<ztol]
        centroid = list(np.average(slice,0))
        line.append(centroid)

    return line

def rotate_about_x_axis(dat, th):
    dat_straight = []
    for row in dat:
        zd = row[2] * math.cos(th) + row[1] * math.sin(th)
        yd = -row[2] * math.sin(th) + row[1] * math.cos(th)
        dat_straight.append([row[0], yd, zd])
    return dat_straight


def repeat_points_along_axis(points, addZEndCentroids=0, plot=1):
    # for input points, identify the z-values where most points are clustered, and the z val where they end (sparse data). Repeat the cluster along the z axis 2/3 times depending on how spaced apart they are, and how long the total zaxis is.
    ztol = 0.9
    z_vals = set(points[:,2])
    zends = [min(z_vals), max(z_vals)]

    if True: # only one repetition of the cluster
        clusterCentroid = np.average(points, 0)
        clusterPoints = [p for p in points if abs(p[2] - clusterCentroid[2]) < ztol]
        zEndNearCluster = zends[0] if abs(zends[0] - clusterCentroid[2]) < abs(zends[1] - clusterCentroid[2]) else zends[1]
        zOffset = abs(clusterCentroid[2] - zEndNearCluster)
        translatedClusterPoints = [[c[0], c[1], c[2]+zOffset] for c in clusterPoints]
    else: # multiple repeats of the cluster along finer z increments
        zreps = 20
        zvals = np.linspace(zends[0]+0.2,zends[1]-0.2,zreps)
        clusterPoints = [p for p in points if abs(p[2] - 5) < ztol]
        for count, iz in enumerate(zvals):
            translatedClusterPoints.extend([[c[0], c[1]-(count*0.02), iz] for c in clusterPoints])

    # add 'ghost' points at centroids of either zend to force ellipse to go there: and increase weighting of these points.
    zEndPoints = [[p for p in points if abs(p[2] - zends[0]) < ztol],
                  [p for p in points if abs(p[2] - zends[1]) < ztol]]
    if addZEndCentroids:
        zEndCentroids = list([np.average(ze,0) for ze in zEndPoints])
        zEndCentroids += zEndCentroids*1
    else:
        zEndCentroids = []

    # double weightings of points at either zends by doubling the number of points.
    if False:
        doublePoints = zEndPoints[0]+zEndPoints[1]
    else:
        doublePoints = []

    if addZEndCentroids: # don't include oriinal points at extreme z locations
        newPoints = np.array(translatedClusterPoints + doublePoints + zEndCentroids)
    else:
        newPoints = np.array(list(points) + translatedClusterPoints + doublePoints + zEndCentroids)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(points[:,0],points[:,1],points[:,2],marker='.',color='r')
        ax.set_title('Original N='+str(len(points)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(newPoints[:,0],newPoints[:,1],newPoints[:,2],marker='.',color='g')
        ax.set_title('extended N='+str(len(newPoints)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

    return newPoints


def compress_points_along_axis(points, addZEndCentroids=0, plot=1):
    # find centroid of whole data.
    # then move z values closer to this centroid depending on side of centroid they are on.
    # add a centroid point for the z centroid of original zslice points.
    ztol = 0.9
    z_vals = set(points[:,2])
    zends = [min(z_vals), max(z_vals)]

    translatedPoints = []
    centroid = np.average(points, 0)
    zinc = 0.2*(zends[1]-zends[0])
    for p in points:
        sign = 1 if p[2]<centroid[2] else -1
        row = [p[0],p[1],p[2]+(sign*zinc)]
        translatedPoints.append(row)
    # translatedPoints = [[c[0], c[1], c[2]+zOffset] for c in clusterPoints]

    # add 'ghost' points at centroids of either zend to force ellipse to go there: and increase weighting of these points.
    zEndPoints = [[p for p in points if abs(p[2] - zends[0]) < ztol],
                  [p for p in points if abs(p[2] - zends[1]) < ztol]]

    if addZEndCentroids:
        zEndCentroids = list([np.average(ze,0) for ze in zEndPoints])
        zEndCentroids += zEndCentroids*1
        newPoints = np.array(translatedPoints + zEndCentroids)
    else:
        zEndCentroids = []
        newPoints = np.array(list(points) + translatedPoints + zEndCentroids)


    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(points[:,0],points[:,1],points[:,2],marker='.',color='r')
        ax.set_title('Original N='+str(len(points)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(newPoints[:,0],newPoints[:,1],newPoints[:,2],marker='.',color='g')
        ax.set_title('extended N='+str(len(newPoints)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # plt.show()

    return newPoints


def mirror_points_along_2axes(points, plot):
    # duplicate all points and mirror along z axis, with axis at the centroid.
    zvals = set(points[:,2])
    zends = [min(zvals), max(zvals)]
    yvals = set(points[:,1])
    yends = [min(yvals), max(yvals)]

    mirrorPoints = [[c[0], yends[0]+yends[1]-c[1], zends[0]+zends[1]-c[2]] for c in points]

    newPoints = np.array(list(points) + mirrorPoints )

    newPoints = repeat_points_along_axis(newPoints, plot=0)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(points[:,0],points[:,1],points[:,2],marker='.',color='r')
        ax.set_title('Original N='+str(len(points)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(newPoints[:,0],newPoints[:,1],newPoints[:,2],marker='.',color='g')
        ax.set_title('extended N='+str(len(newPoints)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

    return newPoints


def rudimentary_ellipsoid_fit(points, hull, plot):
    # For given points, fit ellipsoid:
    # the points furthest apart form principal axes 1
    # rotate PA1 90deg on a given orthogonal plane = PA2   radius = avg distance of points from this line
    # cross product of PA1 and PA2 is PA3                  radius = avg distance of points from this line

    def find_principal_axes_primary(centre, points, tol):
        z = points[:,2]
        zslices = set(z)
        ends = [[p for p in points if abs(p[2]-min(zslices))<tol],
                [p for p in points if abs(p[2]-max(zslices))<tol]]
        bp = [np.average(ends[i],0) for i in range(2)]

        axis = [abs(bp[1][i] - centre[i]) for i in range(3)]
        radius = np.linalg.norm(axis)
        axis = [a/radius for a in axis]
        return radius, axis, bp

    def find_principal_axes_secondary(centre, points, tol, j):
        centrePoints = [p for p in points if abs(p[2]-centre[2])<tol[1]]

        x =  [c[j] for c in centrePoints]
        maxj = max(x) # radius

        # create a new point to ensure that x/y == 0
        bp = [centre,
              [maxj,centre[1],centre[2] if j == 0 else centre[0],maxj,centre[2]]]

        axis = [abs(bp[1][i] - centre[i]) for i in range(3)]
        radius = abs(maxj-centre[j])
        norm = np.linalg.norm(axis)
        axis = [a/norm for a in axis]
        return radius, axis, bp

    tol = [0.25, 1]
    radii = []
    axes = []
    bps = []
    centre = np.average(points, 0)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for i in range(3): # by centroid of the end zslices. Each z-value doesn't entirely encompass the whole slice, so take a tolerance.
        # if True:
        if i<2:
            radius, axis, bp = find_principal_axes_secondary(centre, points, tol, i)
        else:
            radius, axis, bp = find_principal_axes_primary(centre, points, tol[0]) # z
        radii.append(radius)
        axes.append(axis)
        bps.append(bp)
        # else:
        #     hullpoints = points[hull.vertices, :]
        #     # Naive way of finding the best pair in O(H^2) time if H is number of points on hull
        #     hdist = cdist(hullpoints, hullpoints, metric='euclidean')
        #     bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        #     bp = np.array([hullpoints[bestpair[0]], hullpoints[bestpair[1]]])
        # print(bp)
    if plot:
        bps = np.array(bps)
        ax.scatter(points[:,0], points[:,1], points[:,2], marker='.', color='g')
        try:
            ax.scatter(bps[:,0], bps[:,1], bps[:,2], marker='x', color='r')
            ax.plot(bps[0:2, 0], bps[0:2, 1], bps[0:2, 2], color='r')
        except:
            pass
            print('j=10 instead of plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

    axes = np.array(axes)
    axes = np.linalg.inv(axes)
    return centre, radii, axes


def coordinates_opencmiss_to_list(cache, nodes, coordinates, derv):
    xyzlist = []
    dxyzlist = []
    valid_nodes = []
    ccount = coordinates.getNumberOfComponents()

    # for n in nodelist:
    nodeIter = nodes.createNodeiterator()
    node = nodeIter.next()
    while node.isValid():
        nodeID = node.getIdentifier()
        cache.setNode(node)
        result, v1 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, ccount )
        if result == 1:
            xyzlist.append(v1)
            valid_nodes.append(nodeID)
        if derv:
            result, d1 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, ccount )
            result2, d2 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, ccount )
            result3, d3 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, ccount )
            if result == 1:
                # dxyzlist.append([[d1[i], d2[i], d3[i]] for i in range(len(d1))])
                dxyzlist.append([d1,d2,d3])
        node = nodeIter.next()

    if not dxyzlist:
        dxyzlist = [[[0,0]]*3]*len(xyzlist)

    if derv:
        return valid_nodes, xyzlist, dxyzlist
    else:
        return valid_nodes, xyzlist, []


def zinc_read_exf_file(file, raw_data, derv_present, marker_present, otherFieldNames, groupNames, mesh_dimension):
    from_xml = True if raw_data == 2 else False
    context = Context("Example")
    region = context.getDefaultRegion()
    region.readFile(file)
    if not region.readFile(file):
        print('File not readable for zinc')
    fm = region.getFieldmodule()
    cache = fm.createFieldcache()
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    coords_name = "data_coordinates" if raw_data == 1 else "coordinates"
    coordinates = findOrCreateFieldCoordinates(fm, coords_name)
    otherFields = [findOrCreateFieldFiniteElement(fm, n, 1, component_names=("1"), type_coordinate=False) for n in otherFieldNames]
    mesh = fm.findMeshByDimension(mesh_dimension)

    xyzGroups = []
    if groupNames:
        xyzGroups = {c:[] for c in groupNames}
        for subgroup in groupNames:
            group = fm.findFieldByName(subgroup).castGroup()
            nodeGroup = group.getFieldNodeGroup(nodes)
            if nodeGroup.isValid():
                gnodes = nodeGroup.getNodesetGroup()
                nodeIter = gnodes.createNodeiterator()
                node = nodeIter.next()
                groupSize = 0
                while node.isValid():
                    cache.setNode(node)
                    result, x = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
                    xyzGroups[subgroup].append(x)
                    node = nodeIter.next()
                    groupSize += 1
            else:
                del xyzGroups[subgroup]

    all_node_num, xyz_all, dxyz_single = coordinates_opencmiss_to_list(cache, nodes, coordinates, derv_present)
    for oth in otherFields:
        _, ff, _ = coordinates_opencmiss_to_list(cache, nodes, oth, 0)

    element_list = []
    elementIter = mesh.createElementiterator()
    element = elementIter.next()
    while element.isValid():
        eft = element.getElementfieldtemplate(coordinates, -1)  # assumes all components same
        nodeIdentifiers = getElementNodeIdentifiersBasisOrder(element, eft)
        element_list.append(nodeIdentifiers)
        element = elementIter.next()

    if marker_present:
        # raw_data = True if not derv_present else raw_data
        nodes = fm.findNodesetByName('datapoints') if raw_data else fm.findNodesetByName('nodes')
        marker_names = []
        xyz_marker = []
        marker_nodenum = []
        marker_elemxi = {}
        marker_string = "marker_data" if raw_data == 1 else "marker"
        markerNamesField = fm.findFieldByName(marker_string+"_name")
        markerLocation = fm.findFieldByName(marker_string+"_location")
        hostCoordinates = fm.createFieldEmbedded(coordinates, markerLocation)
        if raw_data and not from_xml:
            coordinates = findOrCreateFieldCoordinates(fm, 'marker_'+coords_name)
        fieldcache = fm.createFieldcache()
        nodeIter = nodes.createNodeiterator()
        node = nodeIter.next()
        while node.isValid():
            fieldcache.setNode(node)
            markerName = markerNamesField.evaluateString(fieldcache)
            if markerName is not None:
                if raw_data:
                    result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
                else:
                    result, x = hostCoordinates.evaluateReal(fieldcache, 3)
                marker_names.append(markerName)
                xyz_marker.append(x)
                marker_nodenum.append(node.getIdentifier())
                element, xi = markerLocation.evaluateMeshLocation(fieldcache, 3)
                if element.isValid():
                    marker_elemxi.update({markerName: {'elementID': element.getIdentifier(), 'xi': xi}})
            node = nodeIter.next()

        return all_node_num, xyz_all, dxyz_single, xyzGroups, element_list, xyz_marker, marker_names, marker_nodenum, marker_elemxi
    else:
        return all_node_num, xyz_all, dxyz_single, xyzGroups


def zinc_find_ix_from_real_coordinates(region, regionName, coordName, emergent=0):
    fm = region.getFieldmodule()
    cache = fm.createFieldcache()
    # nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    datapoints = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    dataNamesField = fm.findFieldByName(regionName)
    coordinates = findOrCreateFieldCoordinates(fm, coordName)
    data_coordinates = findOrCreateFieldCoordinates(fm, "data_coordinates")

    if emergent:
        mesh = fm.findMeshByDimension(3)
        if False:
            mesh = fm.findMeshByDimension(2)
            # mesh = meshOrig.copy()
            is_exterior = fm.createFieldIsExterior()
            is_interior = fm.createFieldNot(is_exterior)
            is_exterior_face_xi1_0 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI1_0))
            is_exterior_face_xi1_1 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI1_1))
            is_exterior_face_xi2_0 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI2_0))
            is_exterior_face_xi2_1 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI2_1))
            is_exterior_face_xi3_0 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI3_0))
            is_exterior_face_xi3_1 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
            mesh.destroyElementsConditional(is_interior)
    else:
        mesh = fm.findMeshByDimension(3)

    found_mesh_location = fm.createFieldFindMeshLocation(data_coordinates, coordinates, mesh)
    found_mesh_location.setSearchMode(found_mesh_location.SEARCH_MODE_NEAREST)
    xi_projected_data = {}
    nodeIter = datapoints.createNodeiterator()
    node = nodeIter.next()
    while node.isValid():
        cache.setNode(node)
        element, xi = found_mesh_location.evaluateMeshLocation(cache, 3)
        if not emergent:
            marker_name = dataNamesField.evaluateString(cache).split('_data')[0]
        else:
            marker_name = dataNamesField.evaluateString(cache)
        print(marker_name)
        if element.isValid():
            addProjection = {marker_name:{"elementID": element.getIdentifier(), "xi": xi,"nodeID": node.getIdentifier()}}
            xi_projected_data.update(addProjection)
            if emergent:
                cache.setMeshLocation(element, xi)
                res, xyz = coordinates.evaluateReal(cache, 3)
                print(xyz)
        node = nodeIter.next()
    return xi_projected_data, found_mesh_location


def zinc_find_embedded_location(region, found_mesh_location, organCoordinateName):
    fm = region.getFieldmodule()
    cache = fm.createFieldcache()
    organCoordinates = fm.findFieldByName(organCoordinateName)

    embeddedOrganField = fm.createFieldEmbedded(organCoordinates, found_mesh_location)
    return embeddedOrganField


def zinc_write_element_xi_marker_file(region, allMarkers, xiNodeInfo, regionD, nodeIdentifierStart, coordinates, outFile=[]):
    fm = region.getFieldmodule()
    if outFile:
        fm.beginChange()
    # if xiNodeInfo['nodeType'] == 'nodes':
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    mesh = fm.findMeshByDimension(3)
    mesh1d = fm.findMeshByDimension(1)
    cache = fm.createFieldcache()

    xiNodeName = findOrCreateFieldStoredString(fm, name=xiNodeInfo['nameStr'])
    xiNodeLocation = findOrCreateFieldStoredMeshLocation(fm, mesh, name="elementxi_location")
    xiNodeTemplate = nodes.createNodetemplate()
    xiNodeTemplate.defineField(xiNodeLocation)
    xiNodeTemplate.defineField(coordinates)
    xiNodeTemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
    xiNodeTemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
    xiNodeTemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
    xiNodeTemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)
    xiMeshGroup = AnnotationGroup(region, ('tracts_xi_elements', None)).getMeshGroup(mesh1d)

    nodeIdentifier = nodeIdentifierStart
    for key in allMarkers:
        xiNodeGroup = findOrCreateFieldGroup(fm, xiNodeInfo['groupName']+'_'+key)
        xiNodePoints = findOrCreateFieldNodeGroup(xiNodeGroup, nodes).getNodesetGroup()
        addxiNode = {"name": key, "xi": allMarkers[key]["xi"]}
        xiNodePoint = xiNodePoints.createNode(nodeIdentifier, xiNodeTemplate)
        xiNodePoint.merge(xiNodeTemplate)
        cache.setNode(xiNodePoint)
        elementID = allMarkers[key]["elementID"]
        element = mesh.findElementByIdentifier(elementID)
        result = xiNodeLocation.assignMeshLocation(cache, element, addxiNode["xi"])
        result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, list(regionD[key]['centre']))
        try:
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, list(regionD[key]['axes'][0]))
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, list(regionD[key]['axes'][1]))
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, list(regionD[key]['axes'][2]))
        except:
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, [1,0,0])
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, [0,1,0])
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, [0,0,1])

        nodeIdentifier += 1

    # write 1D elements between embedded nodes
    if True:
        mesh1d = fm.findMeshByDimension(1)
        elementIdentifier = 55555555
        basis1d = fm.createElementbasis(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        eft1d = mesh1d.createElementfieldtemplate(basis1d)
        elementtemplate = mesh1d.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
        result = elementtemplate.defineField(coordinates, -1, eft1d)
        element = mesh1d.createElement(elementIdentifier, elementtemplate)
        result = element.setNodesByIdentifier(eft1d, [nodeIdentifierStart, nodeIdentifier-1])
        xiMeshGroup.addElement(element)

    if outFile:
        fm.endChange()
        region.writeFile(outFile)

    return region


def find_closest_mesh_node(xtarget, xmesh):
    # index = min(xmesh, key=lambda x:abs(x-xtarget))
    # xtarget = np.array(xtarget)
    # xmesh = np.array(xmesh)
    index = np.array([np.linalg.norm(v) for v in abs(xmesh-xtarget)]).argmin()
    xclosest = xmesh[index]
    return list(xclosest), int(index)


def find_closest_end(xyzp, target):
    norm = 1e6
    end_kept = []
    for count, xyz in enumerate(xyzp):
        vdf = [target[i] - xyz[i] for i in range(3)]
        norm_raw = LA.norm(vdf)
        if norm_raw < norm:
            end_kept = xyz
            index = count
            norm = norm_raw
    return end_kept, index