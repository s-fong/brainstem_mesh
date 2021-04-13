# 25 Jan 21 Original structureFile. Each structure in its own region. added simplex triangle elements in 2D.
# 28 Jan option to replace region datapoints with ellipsoid points: ellipsoid found from Tom Judd code that fits ellipse to datapoints. Returns coeffs, axes, radii
# 29 Jan represent one region as one node (at centre of ellipsoid). Glyph scale and orient it using derivatives, which represent the axes.
# 29 Jan superimpose these ellipsoid points on the simpler meshtemplate.
# 9 Feb  add extra BRN regions not listed in openGL/atlasnames files
# 26 Feb show lines on final cmgui file to represent connections (both local and out of brainstem to other organs)
# 2 Mar colour afferent(blue) and efferent(red) lines between external organ and brainstem
# have an aff&eff node for a given organ/nucGroup so only a max of 2 nodes per object. Otherwise, can reuse derivatives.
# 3 Mar: show nerves/tracts as lines, not single glyphs to fit an ellipsoid to.
# 4 Mar: adding cranial nerves that do not exist from data (e.g. CN X)
# 11 Mar: updated external connectivity table to show with cranial nerves com file
# 15 Mar: extend brainstemEnd of nerve outside of the brainstem mesh, to show nerve's entry/exit to/from brainstem structure
# 22 Mar: find elementxi of nuclear groups within brainstem mesh.
# 9 Apr : find these embedded regions in brainstem_coordinates field
# 12 Apr: writing out embedded brainstem coordinates, and not writing (deformed) coordinates and no raw_data

import time
from ellipsoid_tomjudd import ls_ellipsoid, polyToParams3D, ellipsoid_plot
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
from brainstem_tools import extract_coords_from_opengl, extract_struct_names_opengl, rudimentary_ellipsoid_fit, repeat_points_along_axis, compress_points_along_axis, find_closest_mesh_node, find_closest_end, centroids_of_tract, rotate_about_x_axis, zinc_find_ix_from_real_coordinates, zinc_write_element_xi_marker_file, zinc_find_embedded_location
from opencmiss.zinc.context import Context
from opencmiss.zinc.element import Element, Elementbasis
from opencmiss.zinc.field import Field
from opencmiss.zinc.node import Node
from opencmiss.utils.zinc.field import findOrCreateFieldCoordinates, findOrCreateFieldGroup, findOrCreateFieldNodeGroup, findOrCreateFieldStoredString, findOrCreateFieldStoredMeshLocation
import pandas as pd
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup
from scipy.spatial import ConvexHull, Delaunay


class verticesParams:
    # store all vertices parameters
    def __init__(self, eldata, dataCentre, ex, ellipseCentre):#find_vertices():
        self.dataCentre = np.average(eldata, 0)
        self.dataVectorNorms = [np.linalg.norm(v) for v in abs(eldata - dataCentre)]
        self.idataNormLargest = self.dataVectorNorms.index(max(self.dataVectorNorms))
        self.dataVertex = eldata[self.idataNormLargest]
        self.fitVectorNorms = [np.linalg.norm(v) for v in abs(ex - centre)]
        self.ifitNormLargest = self.fitVectorNorms.index(max(self.fitVectorNorms))
        self.fitVertex = ex[self.ifitNormLargest]

        self.centreDiffNorm = abs(np.linalg.norm(ellipseCentre) - np.linalg.norm(dataCentre)) > 1
        self.vertexPointNorm = self.fitVectorNorms[self.ifitNormLargest] - self.dataVectorNorms[self.idataNormLargest] #abs
        # find the norm between point furthest from centre for ellipsoid vs datapoint NORMALISED
        self.vertexPointNormalised = self.fitVectorNorms[self.ifitNormLargest] / \
                                         self.dataVectorNorms[self.idataNormLargest]


def extend_namelist_LR(group, midlineGroups, list=[], list2=[], list3=[], list4=[]):
    side_present = True if group[:2] == 'L ' or group[:2] == 'R ' else False
    if group not in midlineGroups and not side_present:
        list.extend(['L '+group, 'R '+group])
        list2.extend(['L '+group, 'R '+group])
        list3.extend(['L '+group, 'R '+group])
        list4.extend(['L '+group, 'R '+group])
    else:
        list.append(group)
        list2.append(group)
        list3.append(group)
        list4.append(group)

        # print('Added ', group)
    return list, list2, list3, list4

def abbrev_nuclear_names():
    # for a given name, manually type the abbreviation.
    dct = {
        'NUC AMBIGUUS': 'NA',
        'NUC RETROAMBIGUALIS':'NRA',
        'POSTPYRAMIDAL NUCLEUS, RAPHE':'RAPHE',
        'CENTRAL CANAL':'cs',
        'NUC TRACTUS SOLITARIUS':'NTS',
        'KOLLIKER-FUSE NUC':'KF',
        'DORSAL RESPIRATORY GROUP':'DRG',
        'LATERAL PARABRACHIAL NUCLEUS':'LPB',
        'MEDIAL PARABRACHIAL NUCLEUS':'MPB',
        'PONTINE RESPIRATORY GROUP':'PRG',
        'RETROTRAPEZOID NUCLEUS/PARAFACIAL NUCLEUS':'RTN/PF',
        'BOTZINGER COMPLEX':'Botz',
        'PRE-BOTZINGER COMPLEX':'preBotz',
        'rostral VRG':'rVRG',
        'caudal VRG':'cVRG',
        'VENTRAL RESPIRATORY GROUP':'VRG'
    }
    return dct

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

def nerve_modality():
    # show name type and nucleus involved (abbrev)
    modes = {'GSA':	['SENSORY NUC OF V', 'SPINAL TRIGEMINAL NUCLEUS'],
        'GSE':	['ABDUCENS NUC','HYPOGLOSSAL NUC'],
        'GVA':	['NTS','NA'],
        'GVE':	['SUP SALIVATORY NUC', 'INF SALIVATORY NUC', 'DOR MOTOR NUC, VAGUS'],
        'SSA':	['VESTIBULAR COMPLEX','COCHLEAR NUCLEI'],
        'SVA':	['NTS'],
        'SVE':	['NA', 'FACIAL NUC','MOTOR NUC OF V'] }
    return modes

def create_cranial_nerves(cranialDict_raw, regionD, brainstemCentroid, cranial_nerve_nuclei_list):

    def labelTractEnds(nerve, cranialDict_raw, nuclearEndNames, exitEnd = []):
        dict = {}
        # offset = [5,0,0]
        try:
            for i in range(2):
                dict1 = {}
                # account for possible missing nuclearEndNames. Don't skip over remaining.
                for nucleus in nuclearEndNames:
                    try:
                        dict1[s[i]+nucleus] = list(regionD[s[i] + nucleus])
                    except:
                        print('missing ', nucleus)

                if exitEnd:
                    # nerve is missing, so add a point for the end exiting brainstem
                    dict2 = {'brainstemEnd': exitEnd[i]}
                else:
                    _, ind = find_closest_end(cranialDict_raw[s[i] + nerve], brainstemCentroid)
                    dict2 = {'brainstemEnd':
                           cranialDict_raw[s[i] + nerve][len(cranialDict_raw[s[i] + nerve]) - ind - 1]}
                # artificially make the brainstem end of nerve further away from brainstemCentroid by some arbitrary offset
                if False:
                    sign = [1 if dict2['brainstemEnd'][k] > 0 else -1 for k in range(3)]
                    dict2['brainstemEnd'] = [d+(sign[id]*offset[id]) for id, d in enumerate(dict2['brainstemEnd'])]
                dict1.update(dict2)
                dict[s[i]+nerve] = dict1.copy()
        except:
            print('missing ends for ', nerve)
        return dict

    endsDict = {}

    # make connections for nerve+nuc that have differing root names
    # construct missing nerves based on origin and endpoints if known
    nerves = [key for key in cranial_nerve_nuclei_list.keys()]
    exitEnd = {}
    exitEnd['VAGUS'] = [list(np.average([regionD[s[i] + 'INF OLIVE COMPLEX'], regionD[s[i] + 'INF CEREBELLAR PEDUNCLE']], 0)) for i in range(2)]
    exitEnd['GLOSSOPHARYNGEAL'] = [[e*1.15 for e in exitEnd['VAGUS'][i]] for i in range(2)]
    # XII exits between inf olive and pyramid (the boundary of the medulla). Offset by radius of ellipsoid in x for now.
    hnerve = 'HYPOGLOSSAL'
    exitEnd[hnerve] = [list(regionD[s[i] + 'INF OLIVE COMPLEX']) for i in range(2)]
    for i in range(2):
        sign = -1 if exitEnd[hnerve][i][0]<0 else 1
        exitEnd[hnerve][i][0] = sign* (abs(exitEnd[hnerve][i][0]) + regionD[s[i] + 'INF OLIVE COMPLEX'][0] + 1e-3)

    for nerve in nerves:
        if nerve in list(exitEnd.keys()):
            dict = labelTractEnds(nerve+' NERVE', cranialDict_raw, cranial_nerve_nuclei_list[nerve], exitEnd[nerve])
        else:
            dict = labelTractEnds(nerve+' NERVE', cranialDict_raw, cranial_nerve_nuclei_list[nerve])
        endsDict.update(dict)

    return endsDict


def create_multinuclear_group(regionD, bigName, sub, side, midlineGroups, namelist, tract_namelist, addedObjectAsPoint, nuclear_namelist):

    lens = len(sub)
    xyz = [[sum([regionD[side[i]+sub[isub]]['centre'][k] for isub in range(lens)]) / lens for k in range(3)] for i in range(2)]
    regionD.update({'L ' + bigName: {'centre': np.array(xyz[0])}, 'R ' + bigName: {'centre': np.array(xyz[1])}})
    namelist, tract_namelist, addedObjectAsPoint, nuclear_namelist = extend_namelist_LR(bigName, midlineGroups, namelist, tract_namelist, addedObjectAsPoint, nuclear_namelist)

    return (regionD, namelist, tract_namelist, addedObjectAsPoint, nuclear_namelist)

def plot_ellipsoid_and_points(vx,ex,centre,radii,axes,eldata,dataCentre,title, ax=0):
    first = False
    if not ax: #'First' in title:
        first = True
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
    markersize = 150
    ax.scatter(ex[:, 0], ex[:, 1], ex[:, 2], marker='.', color='b')
    ax.scatter(centre[0], centre[1], centre[2], marker='o', color='b', s=markersize)
    ax.scatter(vx.fitVertex[0], vx.fitVertex[1], vx.fitVertex[2], marker='x', color='b', s=markersize)
    ax.scatter(eldata[:, 0], eldata[:, 1], eldata[:, 2], marker='.', color='r')
    ax.scatter(dataCentre[0], dataCentre[1], dataCentre[2], marker='o', color='r', s=markersize)
    ax.scatter(vx.dataVertex[0], vx.dataVertex[1], vx.dataVertex[2], marker='x', color='r', s=markersize)
    ellipsoid_plot(centre, radii, axes, ax=ax, plot_axes=True, cage_color='b')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    print('fitNorm', vx.fitVectorNorms[vx.ifitNormLargest])
    print('data norm', vx.dataVectorNorms[vx.idataNormLargest])
    if first:
        return fig
    else:
        return []


# ---------------------------------
# MAIN
# ---------------------------------
tic = time.perf_counter()

path = "..\\obj_c_files\\"
structureFile = 'all_structures_pythonic.c' #all_structures_pythonic.c #all_structures.c
cxPath = "..\\connectivity\\"
cxFile = cxPath + "tract_connectivity_intra.csv"
cxexFile = cxPath + "tract_connectivity_external_CN.csv"
structNamePath = '..\\atlasnames_updated.c'

data = {}
deformedBodyScale = 10.2  # scale from unit brainstem_coordinates to deformed data. HARDCODED: from scaffoldmaker cylinderMesh meshEdits

delete_datapoints = True
scale_brainstem_mesh = False
include_cx = True
test_OOB = False
writeOut = True
writeBadFitRegions = False
findNearestRegionNames = False
midbrain_test = False
plotFig = False
findNuclearProjections = True

writeOut = False if findNearestRegionNames else writeOut
if writeOut: print('Writing openCMISS files')

structNames, namelist = extract_struct_names_opengl(structNamePath)
nerveWords = ['NERVE','TRACT','TR','FASC', 'FASCICULUS']
midlineGroups = ['CENTRAL CANAL', 'INF CENTRAL N', 'POSTPYRAMIDAL NUCLEUS, RAPHE']

# subset of names matching BRN figure from paper https://www.sciencedirect.com/science/article/pii/B9780124158047000186
brn_namelist0 = ['CENTRAL CANAL',
                'POSTPYRAMIDAL NUCLEUS, RAPHE',
                'NUC AMBIGUUS',
                'NUC RETROAMBIGUALIS',
                'NUC TRACTUS SOLITARIUS',
                'KOLLIKER-FUSE NUC',
                ]
brn_namelist = []
for name in brn_namelist0:
    brn_namelist,_,_, _ = extend_namelist_LR(name, midlineGroups, brn_namelist)
cranial_nerve_nuclei_list = {
    'TRIGEMINAL': ['SENSORY NUC OF V', 'SPINAL TRIGEMINAL NUCLEUS'],#,'MOTOR NUC OF V'],
    'ABDUCENS': ['ABDUCENS NUC'],
    'FACIAL': ['FACIAL NUC', 'NUC TRACTUS SOLITARIUS', 'SPINAL TRIGEMINAL NUCLEUS'],#'SUP SALIVATORY NUC'],
    'VESTIBULOCOCHLEAR': ['VESTIBULAR COMPLEX', 'COCHLEAR NUCLEI'],
    'GLOSSOPHARYNGEAL': ['NUC AMBIGUUS', 'NUC TRACTUS SOLITARIUS', 'SPINAL TRIGEMINAL NUCLEUS'], #, 'INF SALIVATORY NUC'],
    'VAGUS': ['NUC AMBIGUUS', 'NUC TRACTUS SOLITARIUS', 'DOR MOTOR NUC, VAGUS',
              'SPINAL TRIGEMINAL NUCLEUS'],
    'HYPOGLOSSAL': ['HYPOGLOSSAL NUC'],
}
tract_namelist0 = []
for key in cranial_nerve_nuclei_list.keys():
    for nucleus in cranial_nerve_nuclei_list[key]:
        if nucleus not in tract_namelist0:
            tract_namelist0.append(nucleus)
tract_namelist = []
for name in tract_namelist0:
    tract_namelist,_,_,_ = extend_namelist_LR(name, midlineGroups, tract_namelist)

templateMeshPath = 'scaffoldfitter_output\\'
if False:
    templateMeshFileName = 'geofit3_668dipped_c5s3.exf' # without body_coordinates
else:
    templateMeshFileName = 'geofit2_6612_bodycoordinates.exf' # with body_coordinates
templateMeshFile = templateMeshPath+templateMeshFileName
outFileName = 'ellGlyphs_cxlines_' + templateMeshFileName
# outFileName = ['ellipsoidGlyphs' + templateMeshFileName,
#                'BRNsubset_ellipsoidGlyphs' + templateMeshFileName]
outFile = templateMeshPath + 'processed\\' + outFileName

outline = False
data = extract_coords_from_opengl(path, structureFile, outline, data, structNames, wantNorm = 0)

brainstemCentroid = np.average(data['brainSkin']['xyz'],0)
s = ['L ','R ']

# ####################################
# add extra regions (for BRN and cranial nerves)
# ####################################
addedObjectAsPoint = []

# ############ DRG: within NTS
NTSxyz = [data['L NUC TRACTUS SOLITARIUS']['xyz'], data['R NUC TRACTUS SOLITARIUS']['xyz']]
DRGxyz = [np.average(dnts,0) for dnts in NTSxyz]
DRGname = 'DORSAL RESPIRATORY GROUP'
data.update({'L '+DRGname: {'xyz':list(DRGxyz[0])}, 'R '+DRGname: {'xyz':list(DRGxyz[1])}})
namelist, brn_namelist, addedObjectAsPoint,_ = extend_namelist_LR(DRGname, midlineGroups,namelist, brn_namelist, addedObjectAsPoint)
# ############ parabrachial nuclei: surround superior cerebellar peduncle. Between kolliker-fuse and dorsal tegmental nucleus
dPB = 0.1
LPBxyz = [data['L SUP CEREBELLAR PEDUNCLE']['xyz'], data['R SUP CEREBELLAR PEDUNCLE']['xyz']]
MPBxyz = LPBxyz[:]#.copy()
SCPcentroid = [np.average(xyz,0) for xyz in LPBxyz]
LPBxyz = [[x for x in LPBxyz[i] if x[1] > SCPcentroid[i][1]*1.2 and x[2] > SCPcentroid[i][2]] for i in range(2)]
LPBxyz = [LPBxyz[i] + [[p[0]-(dPB*(i==0)) +(dPB*(i==1)),
                        p[1]+dPB, p[2]] for p in LPBxyz[i]] for i in range(2)]
MPBxyz = [[x for x in MPBxyz[i] if x[1] < SCPcentroid[i][1]*1 and x[2] > SCPcentroid[i][2]] for i in range(2)]
MPBxyz = [MPBxyz[i] + [[p[0]+(dPB*(i==0)) -(dPB*(i==1)),
                        p[1]-dPB, p[2]] for p in MPBxyz[i]] for i in range(2)]
LPBname = 'LATERAL PARABRACHIAL NUCLEUS'
MPBname = 'MEDIAL PARABRACHIAL NUCLEUS'
data.update({'L '+LPBname: {'xyz':list(LPBxyz[0])}, 'R '+LPBname: {'xyz':list(LPBxyz[1])}})
data.update({'L '+MPBname: {'xyz':list(MPBxyz[0])}, 'R '+MPBname: {'xyz':list(MPBxyz[1])}})
namelist, brn_namelist, _,_ = extend_namelist_LR(LPBname, midlineGroups, namelist, brn_namelist)
namelist, brn_namelist, _,_ = extend_namelist_LR(MPBname, midlineGroups, namelist, brn_namelist)
# ############ PRG PONTINE RESPIRATORY GROUP PLACEHOLDER POINT
PRGname = 'PONTINE RESPIRATORY GROUP'
KFname = 'KOLLIKER-FUSE NUC'
KFxyz = [np.average(data[s[i]+KFname]['xyz'],0) for i in range(2)]
PRGxyz = [[(np.average([x[k] for x in LPBxyz[i]],0) +
            np.average([x[k] for x in MPBxyz[i]],0) +
            KFxyz[i][k])/3 for k in range(3)] for i in range(2)]
data.update({'L '+PRGname: {'xyz':list(PRGxyz[0])}, 'R '+PRGname: {'xyz':list(PRGxyz[1])}})
namelist, brn_namelist, addedObjectAsPoint,_ = extend_namelist_LR(PRGname, midlineGroups, namelist, brn_namelist, addedObjectAsPoint)
# ############ retrotrapezoid nucleus: thin  sheet under the facial nucleus
dRTN = 0.2
RTNxyz = [data['L FACIAL NUC']['xyz'], data['R FACIAL NUC']['xyz']]
FNcentroid = [np.average(xyz,0) for xyz in RTNxyz]
RTNxyz = [[x for x in RTNxyz[i] if abs(x[0]) > abs(FNcentroid[i][1])*1.5] for i in range(2)]
RTNxyz = [RTNxyz[i] + [[p[0]-(dRTN*(i==0))+(dRTN*(i==1)), p[1], p[2]] for p in RTNxyz[i]] for i in range(2)]
RTNname = 'RETROTRAPEZOID NUCLEUS/PARAFACIAL NUCLEUS'
data.update({'L '+RTNname: {'xyz':list(RTNxyz[0])}, 'R '+RTNname: {'xyz':list(RTNxyz[1])}})
namelist, brn_namelist, _,_ = extend_namelist_LR(RTNname, midlineGroups, namelist, brn_namelist)
# ############ VENTRAL RESPIRATORY COLUMN:
# BEGIN: level of rostral end of NUCLEUS AMBIGUUS (more ventral) (Botz here)
# END:   rostral end of NUCLEUS RETROAMBIGUUS (cVRG here)
VRCname = ['BOTZINGER COMPLEX', 'PRE-BOTZINGER COMPLEX', 'rostral VRG', 'caudal VRG']
VRCxyz = []
NAxyz = [data['L NUC AMBIGUUS']['xyz'], data['R NUC AMBIGUUS']['xyz']]
NRAxyz = [data['L NUC RETROAMBIGUALIS']['xyz'], data['R NUC RETROAMBIGUALIS']['xyz']]
dNA = 0.3
startVRC = [0,0]
endVRC = [0,0]
xyzOffset = [0,0]
ztol = 0.2
for i in range(2):
    npx = [np.array(NAxyz[i])[:, k] for k in range(3)]
    nrpx = [np.array(NRAxyz[i])[:, k] for k in range(3)]
    if i == 0:
        ix = [np.where(npx[k]==min(npx[k]))[0][0] for k in range(3)]
    else:
        ix = [np.where(npx[0] == max(npx[0]))[0][0],
              np.where(npx[1] == min(npx[1]))[0][0],
              np.where(npx[2] == min(npx[2]))[0][0]]
    startVRC[i] = [NAxyz[i][ix[0]][0] - (dNA*(i==0)) + (dNA*(i==1)),
                   NAxyz[i][ix[1]][1] - (dNA*(i==0)) + (dNA*(i==1)),
                   NAxyz[i][ix[2]][2]]
    zlevel = min([x[2] for x in NRAxyz[i]])
    endVRC[i] = np.average([n for n in NRAxyz[i] if abs(n[2]-zlevel)<ztol],0)
    endVRC[i][2] *= 1.1
    xyzOffset[i] = [np.linspace(startVRC[i][k],endVRC[i][k],4) for k in range(3)]
    VRCxyz.append([[xyzOffset[i][k][g] for k in range(3)] for g in range(4)])
for p, pname in enumerate(VRCname):
    data.update({'L '+pname: {'xyz':list(VRCxyz[0][p])}, 'R '+pname: {'xyz':list(VRCxyz[1][p])}})
    namelist, brn_namelist, addedObjectAsPoint,_ = extend_namelist_LR(pname, midlineGroups, namelist, brn_namelist, addedObjectAsPoint)
# ############ VRG VENTRAL RESPIRATORY GROUP PLACEHOLDER POINT
VRGname = 'VENTRAL RESPIRATORY GROUP'
VRGxyz = [[(VRCxyz[i][2][k] + VRCxyz[i][3][k])/2 for k in range(3)] for i in range(2)]
data.update({'L '+VRGname: {'xyz':list(VRGxyz[0])}, 'R '+VRGname: {'xyz':list(VRGxyz[1])}})
namelist, brn_namelist, addedObjectAsPoint,_ = extend_namelist_LR(VRGname, midlineGroups, namelist, brn_namelist, addedObjectAsPoint)

# ############ TRIGEMINAL MOTOR NUCLEUS: near principal sensory nucleus of V, but closer to midline
templatename = 'SENSORY NUC OF V'
templatexyz = [data['L '+templatename]['xyz'], data['R '+templatename]['xyz']]
# newxyz = [np.average(xyz,0) for xyz in templatexyz]
newxyz = [[[t[0]*0.7,t[1],t[2]] for t in row] for row in templatexyz]
newname = 'MOTOR NUC OF V'
data.update({'L '+newname: {'xyz':list(newxyz[0])}, 'R '+newname: {'xyz':list(newxyz[1])}})
namelist, tract_namelist, _,_ = extend_namelist_LR(newname, midlineGroups,namelist, tract_namelist)
cranial_nerve_nuclei_list['TRIGEMINAL'].append(newname)
# ############ SUPERIOR SALIVATORY NUCLEUS: near facial motor, but closer to midline, and smaller. Add as point
templatename = 'FACIAL NUC'
templatexyz = [data['L '+templatename]['xyz'], data['R '+templatename]['xyz']]
newxyz = [np.average(xyz,0) for xyz in templatexyz]
newxyz = [[row[0]*0.7,row[1],row[2]] for row in newxyz]
newname = 'SUP SALIVATORY NUC'
data.update({'L '+newname: {'xyz':list(newxyz[0])}, 'R '+newname: {'xyz':list(newxyz[1])}})
namelist, tract_namelist, addedObjectAsPoint,_ = extend_namelist_LR(newname, midlineGroups,namelist, tract_namelist, addedObjectAsPoint)
cranial_nerve_nuclei_list['FACIAL'].append(newname)
# ############ INFERIOR SALIVATORY NUCLEUS: caudal to superior salivary (assume same 12plane. Add as point
templatename = 'SUP SALIVATORY NUC'
templatexyz = [data['L '+templatename]['xyz'], data['R '+templatename]['xyz']]
newxyz = [[row[0],row[1],row[2]+0.3] for row in templatexyz]
newname = 'INF SALIVATORY NUC'
data.update({'L '+newname: {'xyz':list(newxyz[0])}, 'R '+newname: {'xyz':list(newxyz[1])}})
namelist, tract_namelist, addedObjectAsPoint,_ = extend_namelist_LR(newname, midlineGroups,namelist, tract_namelist, addedObjectAsPoint)
cranial_nerve_nuclei_list['GLOSSOPHARYNGEAL'].append(newname)

#########################
# find regions close to or on the mesh outline (brainSkin.xyz) uasing their datapoints
# add to manualOOBList
#########################
dataMesh = data['brainSkin']['xyz']
brainstemHull = Delaunay(dataMesh)
dataMesh = np.array([[d/deformedBodyScale for d in row] for row in dataMesh])
btol = 2
for region in namelist:
    # print('checking if ',region,' is on boundary')
    regionXYZ = data[region]['xyz']
    if region in addedObjectAsPoint:
        regionXYZ = [regionXYZ]
    # find point furthest from centroid
    index = np.array([np.linalg.norm(v) for v in abs(brainstemCentroid-regionXYZ)]).argmax()
    point = regionXYZ[index]

#########################
# find rotation angle about x axis that data slices are offset by
# use a single slice of an object, manually chosen.
#########################
chosenRegion = 'L ABDUCENS NERVE'
dat = data[chosenRegion]['xyz']
dat = np.array([d for d in dat if d[1] < -3 ])
num = max(dat[:,2]) - min(dat[:,2])
denom = max(dat[:,1]) - min(dat[:,1])
th = math.atan(num/denom)
dat_straight = rotate_about_x_axis(dat, th)

#########################
# fit ellipsoid to datapoints
# for all objects, except tracts
#########################
regionD = {}
nerve_dict = {}
badFitRegions = []
manualBadFitList = ['CENTRAL CANAL', 'R LAT VESTIBULAR NUC (VEN DIV)']
manualBadFitList = []
BFRcount = 0
vertexPointNormaliseds = {}
compressedRegions = []
for region in namelist:
    eldata = np.array(data[region]['xyz'])
    if not any(item in region.split(' ') for item in nerveWords) or 'NUC' in region:
        outsideBoundary = True
        centreOffset = [0.05, 0.08, 0.05] #for outside_boundary
        if region in addedObjectAsPoint:
            regionD.update({region: {'centre': eldata, 'axes': None, 'radii': radii, 'datapoints': [eldata]}})
        else:
            # if region == 'R MEDIAL PARABRACHIAL NUCLEUS':
            #     plotFig = True
            # TAKE CONVEX HULL OF POINTS IN 3D
            hullV = ConvexHull(eldata)
            lH = len(hullV.vertices)
            hull = np.zeros((lH, 3))
            for i in range(len(hullV.vertices)):
                hull[i] = eldata[hullV.vertices[i]]
            hull = np.transpose(hull)
            eansa = ls_ellipsoid(hull[0], hull[1], hull[2])
            centre, radii, axes, R = polyToParams3D(eansa, False)
            # convert points to original untransformed coordinates
            ellipsoidPointsTransformed = [[radii[0],0,0], [0,radii[1],0], [0,0,radii[2]],
                                          [-radii[0], 0, 0], [0, -radii[1], 0], [0, 0, -radii[2]],
                                          [0,0,0] ]
            Q = axes.T
            ellipsoidPoints = [np.dot(Q, row) for row in ellipsoidPointsTransformed]
            ex = np.array([[row[i]+centre[i] for i in range(3)] for row in ellipsoidPoints])
            dataCentre = np.average(eldata,0)
            centre = dataCentre.copy()
            vx = verticesParams(eldata, dataCentre, ex, centre)
            assessFit = vx.vertexPointNormalised > 1.5 or \
                        vx.vertexPointNormalised < 0.67 # vertexPointNorm > 2
            # assessFit = False
            # print('Region: ',region, ' - First fit vertex norm diff: ',vx.vertexPointNorm)
            vertexPointNormaliseds.update({region:vx.vertexPointNormalised})
            if plotFig and assessFit:
                ptitle = region + ' First fit ' + 'normnorm=%2.3f'%(vx.vertexPointNormalised)
                fig = plot_ellipsoid_and_points(vx,ex,centre,radii,axes,eldata,dataCentre,ptitle)
                print('Normalised vertex norm diff', vx.vertexPointNormalised)
                # plt.show()

            if assessFit: # or region in manualBadFitList:
                BFRcount += 1
                # print('Bad fit region',BFRcount,': ', region)
                badFitRegions.append(region)
                if region in manualBadFitList and False: # these are particularly bad fits
                        # # rudimentary fit of ellipsoid to points furthest apart (the PAs)
                    centre, radii, axes = rudimentary_ellipsoid_fit(eldata, hullV, plot=0)
                else:
                    # LS-fit to (unhulled) data
                    compresspoints = False
                    eldata = repeat_points_along_axis(eldata, plot=0)
                    eansa = ls_ellipsoid(eldata[:,0], eldata[:,1], eldata[:,2])
                    centre, radii, axes, R = polyToParams3D(eansa, False)
                    # recalculate and refactor
                    ellipsoidPointsTransformed = [[radii[0], 0, 0], [0, radii[1], 0], [0, 0, radii[2]],
                                                  [-radii[0], 0, 0], [0, -radii[1], 0], [0, 0, -radii[2]],
                                                  [0, 0, 0]]
                    Q = axes.T
                    ellipsoidPoints = [np.dot(Q, row) for row in ellipsoidPointsTransformed]
                    ex = np.array([[row[i] + centre[i] for i in range(3)] for row in ellipsoidPoints])
                    dataCentre = np.average(eldata, 0)
                    vx2 = verticesParams(eldata, dataCentre, ex, centre)
                    worseFit = vx2.vertexPointNormalised > vx.vertexPointNormalised
                    if worseFit:
                        eldata = compress_points_along_axis(eldata, addZEndCentroids = 1, plot=0)
                        eansa = ls_ellipsoid(eldata[:, 0], eldata[:, 1], eldata[:, 2])
                        centre, radii, axes, R = polyToParams3D(eansa, False)
                        compresspoints = True
                        compressedRegions.append(region)

                centre = np.average(eldata, 0).copy()

            #--------------------------
            # check if ellipsoid is OOB
            # repeat incremental translation until no longer OOB
            #--------------------------
            if test_OOB:
                stillOOB = True
                OOBiters = 0
                if True and region not in manualBadFitList:
                    while outsideBoundary and OOBiters < 10: #stillOOB
                        index = np.array([np.linalg.norm(v) for v in abs(brainstemCentroid-ex)]).argmax()
                        point = ex[index]
                        pnorm = np.linalg.norm([point[k]-brainstemCentroid[k] for k in range(3)])
                        if not brainstemHull.find_simplex(point)>=0:
                            outsideBoundary = True
                        else:
                            xNear,_ = find_closest_mesh_node(np.array(point),dataMesh)
                            bnorm = np.linalg.norm([point[k]-xNear[k] for k in range(3)])
                            outsideBoundary = True if bnorm>btol else False

                        # translate points if OOB
                        if outsideBoundary:#regionStandard in manualOOBList:
                            OOBiters += 1
                            if OOBiters == 1:
                                # sign = [1 if centre[k]<brainstemCentroid[k] else -1 for k in range(3)]
                                sign = [1 if point[k]<brainstemCentroid[k] else -1 for k in range(3)]
                            if region in brn_namelist or True:
                                print(region,' outside brainskin')
                            centre = [centre[k] + (sign[k]*centreOffset[k]) for k in range(3)]
                            ex = np.array([[row[i] + centre[i] for i in range(3)] for row in ellipsoidPoints])

            # check fit's CENTRE is within brainstem boundary
            outsideCentres = []
            if not brainstemHull.find_simplex(centre) >= 0:
                print('******')
                print(region,'\'s centre is outside boundary')
                print('******')
                outsideCentres.append(region)

            # recalculate and refactor
            ellipsoidPointsTransformed = [[radii[0],0,0], [0,radii[1],0], [0,0,radii[2]],
                                          [-radii[0], 0, 0], [0, -radii[1], 0], [0, 0, -radii[2]],
                                          [0,0,0] ]
            Q = axes.T
            ellipsoidPoints = [np.dot(Q, row) for row in ellipsoidPointsTransformed]
            ex = np.array([[row[i]+centre[i] for i in range(3)] for row in ellipsoidPoints])
            dataCentre = np.average(eldata,0)
            vx2 = verticesParams(eldata, dataCentre, ex, centre)
            worseFit = vx2.vertexPointNormalised > vx.vertexPointNormalised
            if assessFit and plotFig: #compresspoints:# or (assessFit or region == 'CENTRAL CANAL')) and plotFig: #and worseFit
                pstr = 'COMPRESSED' if compresspoints else 'HACKED'
                ptitle = ' *%s* '%pstr + 'normnorm = %2.3f'%(vx2.vertexPointNormalised)
                ax = fig.add_subplot(122,projection='3d')
                _ = plot_ellipsoid_and_points(vx2, ex, centre, radii, axes, eldata, dataCentre, ptitle, ax)
                # plt.show()
            # if fit is better, keep it. Otherwise, revert to original fit?
            vx = vx2

            regionD.update({region: {'centre': centre, 'axes': ellipsoidPoints[:3], 'radii': radii, 'datapoints': eldata}})
    else:
        # object is a pathway of fibres, to represent as a line element
        eldata_straight = rotate_about_x_axis(eldata, th)
        line_rot = centroids_of_tract(np.array(eldata_straight))
        line = rotate_about_x_axis(np.array(line_rot), -th)
        line = [[l/deformedBodyScale for l in row] for row in line]
        nerve_dict.update({region:line})
        regionD.update({region:{'datapoints':eldata}})
        if False:
            line = np.array(line)
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter(eldata[:,0], eldata[:,1], eldata[:,2],'.')
            ax.scatter(eldata_straight[:,0], eldata_straight[:,1], eldata_straight[:,2],'.')
            ax.plot(line[:,0], line[:,1], line[:,2],'.-')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(region)
            plt.show()

if plotFig:
    plt.show()
print('***** number of out-of-bounds centres = ' + str(len(outsideCentres)) + ' *****')
print('compressed-z regions',compressedRegions)

# find largest area
nuclear_namelist = [n for n in namelist if n not in list(nerve_dict.keys())]
brn_nuclear_namelist = [n for n in brn_namelist if n not in list(nerve_dict.keys())]
tract_nuclear_namelist = [n for n in tract_namelist if n not in list(nerve_dict.keys())]
radiis = [max(regionD[key]['radii']) for key in nuclear_namelist]
largestRegion = namelist[radiis.index(max(radiis))]
print('Largest region: ',largestRegion)

if findNearestRegionNames:
    normTol = 3
    possibleRegions = []
    # do calculations on coordinates of regions i.e. find regions that are coincident with literature on other BRN regions not listed in atlasnames.c
    knownRegionName = 'L FACIAL NUC'
    knownCentre = regionD[knownRegionName]['centre']
    for region in namelist:
        if region != knownRegionName:
            norm = np.linalg.norm([knownCentre[i] - regionD[region]['centre'][i] for i in range(3)])
            # print(region,': ', norm)
            if norm < normTol:
                possibleRegions.append(region)
    print('CLOSE TO ', knownRegionName, ':')
    print(possibleRegions)

# find groups in midbrain region to help determine if oculo/troch nuclei can be shown
if midbrain_test:
    midbrainNucleiZVal = {}
    for region in regionD.keys():
        try:
            if regionD[region]['centre'][2] < -11:
                midbrainNucleiZVal.update({region:regionD[region]['centre'][2]})
        except:
            pass

# ############ add missing cranial nuclei: parse through list to check both L and R are present?
# ############ L ABDUCENS NUC // L LAT VESTIBULAR NUC (VEN DIV)
missing_object_list = []
sideless_namelist = [n[2:] for n in namelist if n not in midlineGroups]
for name in sideless_namelist:
    missing_object = []
    Rsided_name = 'R ' + name
    Lsided_name = 'L ' + name
    if Rsided_name not in namelist:
        missing_object_list.append(Rsided_name)
        missing_object = Rsided_name
    elif Lsided_name not in namelist:
        missing_object_list.append(Lsided_name)
        missing_object = Lsided_name

    # for now, let mirrored points have the exact same points as present side, but signs changed for x values.
    if missing_object:
        xyzMirrored = regionD[Rsided_name if missing_object == Lsided_name else Lsided_name]['centre'].copy()
        dsMirrored = regionD[Rsided_name if missing_object == Lsided_name else Lsided_name]['axes'].copy()
        xyzMirrored[0] *= -1
        dsMirrored = [[row[0]*-1, row[1], row[2]] for row in dsMirrored]
        regionD.update({missing_object: {'centre': np.array(xyzMirrored),
                                         'axes':dsMirrored}})
        namelist, tract_nuclear_namelist,nuclear_namelist,_ = extend_namelist_LR(missing_object, midlineGroups, namelist, tract_nuclear_namelist, nuclear_namelist)
        print('Added ', missing_object)

# ############ group together large groups of nuclei into one average location using subnuclei
complexNames = {'SPINAL TRIGEMINAL NUCLEUS':
                    ['CAUDAL NUC, SPINAL TR OF V', 'INTERPOLAR NUC, SPINAL TR OF V', 'ORAL NUC, SPINAL TR OF V'],
                'COCHLEAR NUCLEI':
                    ['A-V COCHLEAR NUC', 'DOR COCHLEAR NUC', 'P-V COCHLEAR NUC'],
                'VESTIBULAR COMPLEX':
                    ['INF VESTIBULAR NUC', 'MED VESTIBULAR NUC', 'LAT VESTIBULAR NUC (DOR DIV)', 'LAT VESTIBULAR NUC (VEN DIV)', 'SUP VESTIBULAR NUC (MED DIV)', 'SUP VESTIBULAR NUC (LAT DIV)']}

for complexName in complexNames.keys():
    if complexName not in namelist:
        try:
            regionD, namelist, tract_namelist, addedObjectAsPoint, nuclear_namelist = create_multinuclear_group(regionD, complexName, complexNames[complexName], s, midlineGroups, namelist, tract_namelist, addedObjectAsPoint, nuclear_namelist)
        except:
            pass


####################################################################################################
# ZINC    ZINC    # ZINC    ZINC    # ZINC    ZINC    # ZINC    ZINC    # ZINC    ZINC
####################################################################################################
if writeOut:

    noRawData = True # don't write out raw_data
    noRawData = True if delete_datapoints else noRawData

    #########################
    # read connectivity files (csv)
    #########################
    cx_df = pd.read_csv(cxFile, index_col=0, encoding='latin1')
    sourcesIntra = list(cx_df.columns)
    targetsIntra = list(cx_df.index)

    cxex_df = pd.read_csv(cxexFile, index_col=0, encoding='latin1')
    nucleiEx = list(cxex_df.index)
    externalOrganEx = list(cxex_df.columns)
    externalOrganList = externalOrganEx
    midlineGroups += externalOrganList

    if writeBadFitRegions:
        namelist = badFitRegions
        outFileName = 'badFitSubset_template_with_ellipsoidGlyphs.exf'
        outFile = templateMeshPath + 'processed\\' + outFileName
        print('Computing for badly fit regions')

    #########################
    # opencmiss - write to ex
    #########################
    context = Context("Example")
    region = context.getDefaultRegion()
    region.readFile(templateMeshFile)
    fm = region.getFieldmodule()
    fm.beginChange()
    oldChild = region.findChildByName('raw_data')
    region.removeChild(oldChild)
    regionName = findOrCreateFieldStoredString(fm, name="brainstem_region_name")
    dpoints = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    dcoordinates = findOrCreateFieldCoordinates(fm, "data_coordinates")
    # else:
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    coordinates = findOrCreateFieldCoordinates(fm, "coordinates")
    nodeType = 'nodes'
    ccount = coordinates.getNumberOfComponents()
    nodetemplate = nodes.createNodetemplate()
    nodetemplate.defineField(coordinates)
    nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
    nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
    nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
    nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)
    nodetemplate.defineField(regionName)
    dnodetemplate = dpoints.createNodetemplate()
    dnodetemplate.defineField(dcoordinates)
    dnodetemplate.setValueNumberOfVersions(dcoordinates, -1, Node.VALUE_LABEL_VALUE, 1)
    dnodetemplate.defineField(regionName)

    BRNGroup = findOrCreateFieldGroup(fm, 'BRN group')
    BRNPoints = findOrCreateFieldNodeGroup(BRNGroup, nodes).getNodesetGroup()
    tractGroupAll = findOrCreateFieldGroup(fm, 'all tracts and nuclei')
    tractPointsAll = findOrCreateFieldNodeGroup(tractGroupAll, nodes).getNodesetGroup()
    mesh3d = fm.findMeshByDimension(3)
    cache = fm.createFieldcache()

    if scale_brainstem_mesh:
        scale = 1.2
        yadj = 0.6
        nodeIter = nodes.createNodeiterator()
        node = nodeIter.next()
        while node.isValid():
            nodeID = node.getIdentifier()
            # print(nodeID)
            cache.setNode(node)
            ds0 = [0]*ccount
            result, x0 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, ccount)
            res, ds0[0] = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, ccount)
            res, ds0[1] = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, ccount)
            res, ds0[2] = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, ccount)
            x1 = [x0[0]*scale, (x0[1]*scale)+yadj, x0[2]*scale]
            ds = [[id * scale for id in drow] for drow in ds0]
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x1)
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, ds[0])
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, ds[1])
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, ds[2])
            node = nodeIter.next()
    nodeOffset = nodes.getSize()
    dnID = 1

    ds1 = [1,0,0]
    ds2 = [0,1,0]
    ds3 = [0,0,1]

    # brnDict = abbrev_nuclear_names()
    ultimateNodeIDdict = {}
    for ni, key in enumerate(nuclear_namelist):
        x = list(regionD[key]['centre'])
        try:
            axes = regionD[key]['axes']
            axes = [ds1, ds2, ds3] if regionD[key]['axes'] == None else axes
        except:
            axes = [ds1, ds2, ds3]
        nodeIdentifier = dnID + ni #nodeOffset + ni + 1
        if not delete_datapoints:
            nuclearPoints = findOrCreateFieldNodeGroup(findOrCreateFieldGroup(fm, 'nuclear group '+key), dpoints).getNodesetGroup()

        dpoint = dpoints.createNode(nodeIdentifier, dnodetemplate)
        try:
            cache.setNode(dpoint)
            dcoordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x)
            # dcoordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, list(axes[0]))
            # dcoordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, list(axes[1]))
            # dcoordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, list(axes[2]))
            ultimateNodeIDdict[key] = nodeIdentifier
        except:
            print('Failed setting node parameters')
        side = key[:2] if key not in midlineGroups else ''
        name = key[2:] if key not in midlineGroups else key
        regionName.assignString(cache, side + name)
        if key in brn_namelist:
            BRNPoints.addNode(dpoint)
        elif key in tract_namelist:
            tractPointsAll.addNode(dpoint)
        if not delete_datapoints:
            nuclearPoints.addNode(dpoint)

    nodeIdentifier = nodes.getSize() + 1
    if not noRawData:#writeBadFitRegions: # or True:
        childRegion = region.createChild('raw_data')
        fmCh = childRegion.getFieldmodule()
        fmCh.beginChange()
        data_coordinates = findOrCreateFieldCoordinates(fmCh, "data_coordinates")
        dnodes = fmCh.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplateraw = dnodes.createNodetemplate()
        nodetemplateraw.defineField(data_coordinates)
        nodetemplateraw.setValueNumberOfVersions(data_coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        cache = fmCh.createFieldcache()

        for name in namelist:
            group = findOrCreateFieldGroup(fmCh, 'group ' + name)
            points = findOrCreateFieldNodeGroup(group, dnodes).getNodesetGroup()
            try:
                xs = regionD[name]['datapoints']
                for x in xs:
                    node = dnodes.createNode(nodeIdentifier, nodetemplateraw)
                    cache.setNode(node)
                    try:
                        data_coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, list(x))
                    except:
                        pass
                    points.addNode(node)
                    nodeIdentifier += 1
            except:
                print('can\'t create a datapoint for ', name)
        fmCh.endChange()

    #########################
    # find projection (elementxi) of nuclear group in mesh
    # nuclear group coordinates are in the region
    #########################
    regionNameStr = regionName.getName()
    if findNuclearProjections:

        br_coordinates_name = 'brainstem_coordinates'
        br_coordinates = findOrCreateFieldCoordinates(fm, br_coordinates_name)
        toc = time.perf_counter()
        print('elapsed time BEFORE finding ix: ', toc - tic, ' s')
        projected_data, found_mesh_location = zinc_find_ix_from_real_coordinates(region, regionNameStr)
        toc = time.perf_counter()
        print('elapsed time AFTER finding ix: ', toc - tic, ' s')

        markerInfo = {}
        xiGroupRootName = 'xiGroups'
        markerInfo['groupName'] = xiGroupRootName
        markerInfo['nameStr'] = regionNameStr
        markerInfo['nodeType'] = nodeType
        nstart = nodes.getSize() + 1
        if False:
            region = zinc_write_element_xi_marker_file(region, projected_data, markerInfo, regionD, nstart, coordinates)

        # find embedded location in brainstem_coordinates using elementxi
        embeddedOrganField = zinc_find_embedded_location(region, found_mesh_location, br_coordinates_name)

        cache = fm.createFieldcache()
        regionD_brainstemCoordinates = {}
        for nucReg in projected_data.keys():
            dpoint = dpoints.findNodeByIdentifier(projected_data[nucReg]['nodeID'])
            cache.setNode(dpoint)
            res, xReal = embeddedOrganField.evaluateReal(cache, 4)
            regionD_brainstemCoordinates.update({nucReg: xReal[:3]})

        # write out to nodes
        embeddedNodeGroup = findOrCreateFieldGroup(fm, 'embedded regions group')
        embeddedPoints = findOrCreateFieldNodeGroup(embeddedNodeGroup, nodes).getNodesetGroup()
        xiNodeLocation = findOrCreateFieldStoredMeshLocation(fm, mesh3d, name="elementxi_location")
        BRregionName = findOrCreateFieldStoredString(fm, name="embedded_region_name")
        BRnodetemplate = nodes.createNodetemplate()
        BRnodetemplate.defineField(br_coordinates)
        BRnodetemplate.setValueNumberOfVersions(br_coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        BRnodetemplate.setValueNumberOfVersions(br_coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        BRnodetemplate.setValueNumberOfVersions(br_coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        BRnodetemplate.setValueNumberOfVersions(br_coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)
        BRnodetemplate.defineField(coordinates)
        BRnodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        BRnodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        BRnodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        BRnodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)
        BRnodetemplate.defineField(xiNodeLocation)
        BRnodetemplate.defineField(BRregionName)

        nodeIdentifier = nodes.getSize() + 1
        for nucReg in regionD_brainstemCoordinates.keys():
            node = nodes.createNode(nodeIdentifier, BRnodetemplate)
            cache.setNode(node)
            xiNodeGroup = findOrCreateFieldGroup(fm, xiGroupRootName + '_' + nucReg)
            xiNodePoints = findOrCreateFieldNodeGroup(xiNodeGroup, nodes).getNodesetGroup()
            result = br_coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, regionD_brainstemCoordinates[nucReg])
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, list(regionD[nucReg]['centre']))
            try:
                axes = regionD[nucReg]['axes']
                axes = [ds1, ds2, ds3] if regionD[nucReg]['axes'] == None else axes
            except:
                axes = [ds1, ds2, ds3]
            result = br_coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, [d/deformedBodyScale for d in axes[0]])
            result = br_coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, [d/deformedBodyScale for d in axes[1]])
            result = br_coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, [d/deformedBodyScale for d in axes[2]])
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, list(axes[0]))
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, list(axes[1]))
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, list(axes[2]))
            elementID = projected_data[nucReg]["elementID"]
            xi = projected_data[nucReg]["xi"]
            element = mesh3d.findElementByIdentifier(elementID)
            result = xiNodeLocation.assignMeshLocation(cache, element, xi)
            BRregionName.assignString(cache, nucReg)
            embeddedPoints.addNode(node)
            xiNodePoints.addNode(node)
            nodeIdentifier += 1

    # remove datapoints - they were only used for found_mesh_location
    if delete_datapoints:
        result = dpoints.destroyAllNodes()

    # Line elements for tracts
    if include_cx:
        coordinates = findOrCreateFieldCoordinates(fm, br_coordinates_name) if findNuclearProjections else findOrCreateFieldCoordinates(fm, "coordinates")
        cache = fm.createFieldcache()
        mesh1d = fm.findMeshByDimension(1)
        elementIdentifier = mesh1d.getSize() + 1
        basis1d = fm.createElementbasis(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        eft1d = mesh1d.createElementfieldtemplate(basis1d)
        elementtemplate = mesh1d.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
        result = elementtemplate.defineField(coordinates, -1, eft1d)

        # show nerve tracts within brainstem
        tractNodeGroup = findOrCreateFieldGroup(fm, 'tract group')
        tractPoints = findOrCreateFieldNodeGroup(tractNodeGroup, nodes).getNodesetGroup()
        tractMeshGroup = AnnotationGroup(region, ('nerve tracts', None)).getMeshGroup(mesh1d)

        cranialNameDict = cranial_nerve_names()
        cranialNerve_names = [cranialNameDict[key] for key in cranialNameDict]
        # some must be created from information from literature...
        # ... some tracts are present in data (nerve_dict), which must be distinct from other non-cranial nerve tracts.
        cranialDict_raw = {}
        nodeIdentifier = nodes.getSize() + 1
        for name in nerve_dict.keys():
            if name[2:].split(' ')[0] not in cranialNerve_names:
                for ix, x in enumerate(nerve_dict[name]):
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                    cache.setNode(node)
                    res = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, ds1)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, ds2)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, ds3)
                    regionName.assignString(cache, name)
                    nodeIdentifier += 1
                    if ix > 0:
                        enodes = [nodeIdentifier - 2, nodeIdentifier - 1]
                        element = mesh1d.createElement(elementIdentifier, elementtemplate)
                        result = element.setNodesByIdentifier(eft1d, enodes)
                        tractMeshGroup.addElement(element)
                        elementIdentifier += 1
                    if ix == len(nerve_dict[name]) - 1:
                        tractPoints.addNode(node)
            else:
                cranialDict_raw.update({name:nerve_dict[name]})

        modeDict = nerve_modality()
        modes = [m for m in modeDict.keys()]
        intraGroupTerm = ('intra-connections', None)
        brainstemGroupTerm = ('brainstem', None)
        intraGroup = AnnotationGroup(region, intraGroupTerm)
        brainstemGroup = AnnotationGroup(region, brainstemGroupTerm)
        intraMeshGroup = intraGroup.getMeshGroup(mesh1d)
        brainstemMeshGroup = brainstemGroup.getMeshGroup(mesh1d)

        ########################
        # global external organs
        ########################
        # for intra connections
        if not cxexFile:
            for source in sourcesIntra:
                sourcename = 'R '+source if source not in midlineGroups else source
                specTargets = [t for t in targetsIntra if cx_df[source][t] == 1]
                for target in specTargets:
                    enodes = [0,0]
                    targetname = 'R '+target if target not in midlineGroups else target
                    try:
                        enodes[0] = ultimateNodeIDdict[sourcename]
                        try:
                            enodes[1] = ultimateNodeIDdict[targetname]
                            element = mesh1d.createElement(elementIdentifier, elementtemplate)
                            result = element.setNodesByIdentifier(eft1d, enodes)
                            intraMeshGroup.addElement(element)
                            elementIdentifier += 1
                        except:
                            print('node doesn\'t exist for ', target)
                    except:
                        print('node doesn\'t exist for ',source)
        exyz = {}
        ey = 0
        ez = max(dataMesh[:,2]) + 2
        labeloffset = [1,1,-0.5]
        exRange = [min(dataMesh[:,0]) - labeloffset[0], max(dataMesh[:,0]) + labeloffset[0]]
        exs = np.linspace(exRange[0], exRange[1], len(externalOrganList))
        zradius = max(abs(exRange[0]),abs(exRange[1])) + 0.025
        ezs = [np.sqrt(zradius**2 - x**2)+labeloffset[2] for x in exs]
        nodetemplateEx = nodes.createNodetemplate()
        nodetemplateEx.defineField(coordinates)
        nodetemplateEx.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        nodetemplateEx.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        dsMag = -1e-1
        externalOrganCoordinates = {}
        for io, organ in enumerate(externalOrganList):
            exyz = [exs[io],ey,ezs[io]]
            node = nodes.createNode(nodeIdentifier, nodetemplateEx)
            cache.setNode(node)
            xadj = (exs[io]/max(exRange)) #abs
            ds = [0,0,dsMag*xadj]
            # the closer to 0 the x value is, the smaller the magnitude.
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, exyz)
            result = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, ds)
            ultimateNodeIDdict.update({organ:nodeIdentifier})
            externalOrganCoordinates[organ] = exyz
            nodeIdentifier += 1

        print('creating external connection elements')

        # cranial nerves and their nuclei
        endsDict = create_cranial_nerves(cranialDict_raw, regionD_brainstemCoordinates, brainstemCentroid, cranial_nerve_nuclei_list)
        # account for several origins (single line element each)
        existingCranialRegions = []
        brainstemEndOffset = [0.5,0,0]
        nucleiEndPointIDs = {}
        nonexistentNodeInConnection = []
        for cn in endsDict.keys():
            # c_elementIdentifier = 1
            cranialNerve = cn[2:].split(' ')[0] if cn not in midlineGroups else cn
            sidestr = cn[:2] if cn not in midlineGroups else ''
            seenOnce = False
            if cranialNerve not in existingCranialRegions:
                seenOnce = True
                childRegion = region.createChild(cranialNerve)
                fmCh = childRegion.getFieldmodule()
                fmCh.beginChange()
                CNname = findOrCreateFieldStoredString(fmCh, name="cranialnerve_object_name")
                child_coordinates = findOrCreateFieldCoordinates(fmCh, br_coordinates_name) #coordinates
                cnodes = fmCh.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
                nodetemplatechild = cnodes.createNodetemplate()
                nodetemplatechild.defineField(child_coordinates)
                nodetemplatechild.setValueNumberOfVersions(child_coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
                nodetemplatechild.setValueNumberOfVersions(child_coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
                nodetemplatechild.setValueNumberOfVersions(child_coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
                nodetemplatechild.setValueNumberOfVersions(child_coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)
                nodetemplatechild.defineField(CNname)
                numDim = 1
                cmesh1d = fmCh.findMeshByDimension(numDim)
                basisChild = fmCh.createElementbasis(numDim, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
                eftChild = cmesh1d.createElementfieldtemplate(basisChild)
                elementtemplateChild = cmesh1d.createElementtemplate()
                elementtemplateChild.setElementShapeType(Element.SHAPE_TYPE_LINE)
                result = elementtemplateChild.defineField(child_coordinates, -1, eftChild)
                ccache = fmCh.createFieldcache()
                existingCranialRegions.append(cranialNerve)
                c_elementIdentifier = 1

            cranialNodeGroup = findOrCreateFieldGroup(fmCh, sidestr+'points')
            cranialPoints = findOrCreateFieldNodeGroup(cranialNodeGroup, cnodes).getNodesetGroup()
            cranialMeshGroup = AnnotationGroup(region, (sidestr+'nerves', None)).getMeshGroup(cmesh1d)

            missing_nerve = False
            nodeIdentifierNerveConnection = []
            if cn in list(cranialDict_raw.keys()) and len(endsDict[cn]) <=2 : # nerve may not be present in cranialDict_raw
                nervePoints = cranialDict_raw[cn]
                if all([endsDict[cn]['brainstemEnd'][k] == nervePoints[0][k] for k in range(3)]):
                    nervePoints.reverse()
                nervePoints = [list(endsDict[cn][key]) for key in endsDict[cn].keys() if 'brainstemEnd' not in key]  + nervePoints
            else:
                # create nerve from endsDict
                missing_nerve = True
                # nuclearPoints = [endsDict[cn][origin] for origin in endsDict[cn] if 'brainstemEnd' not in origin]
                nervePoints = [endsDict[cn]['brainstemEnd']]
            if True:
                xsign = 1 if nervePoints[-1][0] > 0 else -1
                nervePoints.append([nervePoints[-1][0]+xsign*brainstemEndOffset[0], nervePoints[-1][1], nervePoints[-1][2]])
            nodeIdentifiers = []
            for ix, x in enumerate(nervePoints):
                node = cnodes.createNode(nodeIdentifier, nodetemplatechild)
                ccache.setNode(node)
                child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS1, 1, ds1)
                child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS2, 1, ds2)
                child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS3, 1, ds3)

                if ix > 0:
                    nodeIdentifiers.append(nodeIdentifier)
                    if ix == len(nervePoints) - 1:
                        CNname.assignString(ccache, sidestr+'brainstemBoundaryEnd')  # key # ' '
                    else:
                        CNname.assignString(ccache, ' ')
                    enodes = [nodeIdentifier - 1, nodeIdentifier]
                    element = cmesh1d.createElement(c_elementIdentifier, elementtemplateChild)
                    result = element.setNodesByIdentifier(eftChild, enodes)
                    cranialMeshGroup.addElement(element)
                    c_elementIdentifier += 1
                else: # ix == 0
                    if not missing_nerve:
                        CNname.assignString(ccache, [e for e in endsDict[cn].keys() if 'brainstem' not in e][0])
                    else:
                        CNname.assignString(ccache, ' ')
                    nodeIdentifierNerveConnection = nodeIdentifier
                cranialPoints.addNode(node)
                nodeIdentifier += 1
            nucleiEndPointIDs.update({cn:nodeIdentifiers})

            if missing_nerve or len(list(endsDict[cn].keys()))>2:
                for nucleus in endsDict[cn]:
                    if 'brainstemEnd' not in nucleus:
                        try:
                            axes = regionD[nucleus]['axes']
                            axes = [ds1, ds2, ds3] if regionD[nucleus]['axes'] == None else axes
                        except:
                            axes = [ds1, ds2, ds3]
                        cnuclearGroup = findOrCreateFieldGroup(fmCh, 'nuclear group ' + nucleus[2:])
                        cnuclearPoints = findOrCreateFieldNodeGroup(cnuclearGroup, cnodes).getNodesetGroup()
                        node = cnodes.createNode(nodeIdentifier, nodetemplatechild)
                        ccache.setNode(node)
                        child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_VALUE, 1, endsDict[cn][nucleus])
                        child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS1, 1, list(axes[0]))
                        child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS2, 1, list(axes[1]))
                        child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS3, 1, list(axes[2]))
                        CNname.assignString(ccache, nucleus)  # key
                        cranialPoints.addNode(node)
                        cnuclearPoints.addNode(node)
                        try:
                            ultimateNodeIDdict[cranialNerve].update({nucleus: nodeIdentifier})
                        except:
                            ultimateNodeIDdict.update({cranialNerve: {nucleus: nodeIdentifier}})
                        enodes = [nodeIdentifier, nodeIdentifierNerveConnection]
                        element = cmesh1d.createElement(c_elementIdentifier, elementtemplateChild)
                        result = element.setNodesByIdentifier(eftChild, enodes)
                        cranialMeshGroup.addElement(element)
                        c_elementIdentifier += 1
                        nodeIdentifier += 1

            else:
                # add glyph of nuclear point (repeat of nuclearpoints code)
                for key in cranial_nerve_nuclei_list[cn.split(' ')[1]]:  # enumerate(data.keys()):
                    sidedKey = sidestr+key
                    try:
                        # x = list(regionD[sidedKey]['centre'])
                        x = list(regionD_brainstemCoordinates[sidedKey])
                        try:
                            axes = regionD[sidedKey]['axes']
                            axes = [ds1, ds2, ds3] if regionD[sidedKey]['axes'] == None else axes
                        except:
                            axes = [ds1, ds2, ds3]
                    except:
                        print('missing glyph ', sidedKey)
                        pass
                    cnuclearGroup = findOrCreateFieldGroup(fmCh, 'nuclear group ' + key)
                    cnuclearPoints = findOrCreateFieldNodeGroup(cnuclearGroup, cnodes).getNodesetGroup()

                    node = cnodes.createNode(nodeIdentifier, nodetemplatechild)
                    try:
                        ccache.setNode(node)
                        child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                        child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS1, 1, list(axes[0]))
                        child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS2, 1, list(axes[1]))
                        child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS3, 1, list(axes[2]))
                        CNname.assignString(ccache, ' ')#sidestr + key)
                        cnuclearPoints.addNode(node) ######################################
                        try:
                            ultimateNodeIDdict[cranialNerve].update({sidedKey: nodeIdentifier})
                        except:
                            ultimateNodeIDdict.update({cranialNerve:{sidedKey: nodeIdentifier}})
                        nodeIdentifier += 1
                    except:
                        pass
            # ##############################################################################
            # create child  copies of external organs with nodes to make modality connections
            # ##############################################################################
            if seenOnce:
                modeGroups = []
                modeMeshGroups = []
                exZincGroup = findOrCreateFieldGroup(fmCh, 'external organ group')
                modeExPoints = findOrCreateFieldNodeGroup(exZincGroup, cnodes).getNodesetGroup()
                for mode in modes:
                    modeGroups.append(AnnotationGroup(childRegion, ('%s-connections'%mode, None)))
                    modeMeshGroups.append(modeGroups[-1].getMeshGroup(cmesh1d))
                regionExOrganName = findOrCreateFieldStoredString(fmCh, name="external_organ_name")
                nodetemplateEx = cnodes.createNodetemplate()
                nodetemplateEx.defineField(child_coordinates)
                nodetemplateEx.setValueNumberOfVersions(child_coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
                nodetemplateEx.setValueNumberOfVersions(child_coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
                nodetemplateEx.defineField(regionExOrganName)
                for io, organ in enumerate(externalOrganList):
                    exyz = externalOrganCoordinates[organ]
                    node = cnodes.createNode(nodeIdentifier, nodetemplateEx)
                    ccache.setNode(node)
                    xadj = (exs[io]/max(exRange)) #abs
                    ds = [0,0,dsMag*xadj] # if j == 0 else [0,0,dsMag*xadj]
                    # the closer to 0 the x value is, the smaller the magnitude.
                    result = child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_VALUE, 1, exyz)
                    result = child_coordinates.setNodeParameters(ccache, -1, Node.VALUE_LABEL_D_DS1, 1, ds)
                    regionExOrganName.assignString(ccache, organ)#+'-'+modes[j])
                    try:
                        ultimateNodeIDdict[cranialNerve].update({organ:nodeIdentifier})
                    except:
                        ultimateNodeIDdict.update({cranialNerve:{organ:nodeIdentifier}})
                    externalOrganCoordinates[organ] = exyz
                    nodeIdentifier += 1
            # print('creating external connection elements')
            for org in externalOrganEx:
                orgName = 'R ' + org if org not in midlineGroups else org
                linkedNucs = [t for t in nucleiEx if cxex_df[org][t] != '0'] # will have a mode type here.
                for nuc in linkedNucs:
                    nucName = 'R '+nuc if nuc not in midlineGroups else nuc
                    enodes = [0,0]
                    try:
                        enodes[0] = ultimateNodeIDdict[cranialNerve][nucName]
                        try:
                            enodes[1] = ultimateNodeIDdict[cranialNerve][orgName]
                            thisModeAndCN = cxex_df[org][nuc].split('/')
                            for this in thisModeAndCN:  # there may be more than one
                                res = this.split('.')
                                thisMode = res[0]
                                theseCNs = [int(res[i]) for i in range(1, len(res))]
                                imode = modes.index(thisMode)
                                for thisCN in theseCNs:
                                    if cranialNameDict[thisCN] == cranialNerve:
                                        linkExists = True
                                        midNodes = nucleiEndPointIDs['R ' + cranialNameDict[thisCN] + ' NERVE']
                                        totalNodes = []
                                        totalNodes.append(enodes[0])
                                        for mi in midNodes:
                                            totalNodes.append(mi)
                                        totalNodes.append(enodes[1])
                                        for hn in range(1, len(totalNodes)):
                                            element = cmesh1d.createElement(c_elementIdentifier, elementtemplateChild)
                                            result = element.setNodesByIdentifier(eftChild,
                                                                                  [totalNodes[hn - 1], totalNodes[hn]])
                                            c_elementIdentifier += 1
                                            modeMeshGroups[imode].addElement(element)
                            # if linkExists:
                            for nID in enodes:                          ###
                                node = cnodes.findNodeByIdentifier(nID) ###
                                modeExPoints.addNode(node)              ###

                        except:
                            if orgName not in nonexistentNodeInConnection:
                                print('node doesn\'t exist for ', orgName)
                                nonexistentNodeInConnection.append(orgName)
                    except:
                        if nucName not in nonexistentNodeInConnection:
                            print('node doesn\'t exist for ',nucName)
                            nonexistentNodeInConnection.append(nucName)

            # end only if a midline or an R group
            if not seenOnce or cn in midlineGroups:
                fmCh.endChange()

            # fmCh.endChange()

    fm.endChange()
    region.writeFile(outFile)

    if False:
        # copy structureFile into data folder of mapclient workflow
        workflow_path = "C:\\Users\\sfon036\\Google Drive\\SPARC_work\\codes\\mapclient_workflows\\brainstem_outline_fit_1\\data\\" #brainstem_nuclear_groups_only
        shutil.copyfile(outFile[0], workflow_path+outFile[0].split('\\')[-1])

    #----------------------------------------------------
    # write com structureFile to view with all subregions
    #----------------------------------------------------
    coordinates_str = br_coordinates_name if findNuclearProjections else "coordinates"
    cols = ["gold", "silver", "green", "cyan", "orange", "magenta", "yellow", "white", "red"]
    numcols = len(cols)
    rs = 1/deformedBodyScale # reciprocal

    for j in range(3): #len(outFileName)):
        BRN = True if j == 1 else False
        TRACT = True if j == 2 else False
        ALL = (not BRN) * (not TRACT)
        # namelist_no_tracts = [n for n in namelist if not any(item in n.split(' ') for item in nerveWords)]
        datStr = 'data_' if ALL and findNuclearProjections else ''
        coordStr = 'nuclearRegion_coordinates' if findNuclearProjections else 'coordinates' # if ALL
        domainStr = '_datapoints' if ALL and  findNuclearProjections else '_nodes'

        if midbrain_test:
            nuclear_namelist = [m for m in midbrainNucleiZVal.keys()]
        nuclearGrouplist = brn_nuclear_namelist if BRN else (tract_nuclear_namelist if TRACT else nuclear_namelist)
        subgroup_name = ' subgroup "BRN group"' if BRN else (' subgroup "tract group"' if TRACT else '')

        outputComFile = templateMeshPath + 'processed\\' + 'view_'+('BRNsubset_'*BRN)+('TRACTonly_'*TRACT)+('ALLelementxi_'*ALL)+('topMidbrain_'*midbrain_test)+outFileName+'.com'
        with open(outputComFile,'w') as w_out:
            w_out.write('gfx read elements "%s"\n\n'%outFileName)

            if findNuclearProjections:
                w_out.write('gfx define field nuclearRegion_coordinates embedded element_xi elementxi_location field %s\n\n' %(coordinates_str))
            if writeBadFitRegions:
                cols = ['red','green','blue']
                numcols = len(cols)

            for i in range(3):
                w_out.write('gfx define field d%d node_value fe_field %s d/ds%d\n' % (i + 1, coordinates_str,i + 1))
            w_out.write('gfx define field orientation_scale composite d1 d2 d3\n')
            for i in range(3):
                w_out.write('gfx define field dfd%d node_value fe_field coordinates d/ds%d\n' % (i + 1,i + 1))
            w_out.write('gfx define field deformed_orientation_scale composite dfd1 dfd2 dfd3\n\n')
            w_out.write('gfx modify g_element "/" general clear;\n' )
            w_out.write('gfx modify g_element "/" lines domain_mesh1d subgroup brainstem coordinate %s face all tessellation default LOCAL line line_base_size 0 select_on material grey50 selected_material default_selected render_shaded;\n' %(coordinates_str))
            w_out.write('gfx modify g_element "/" surfaces domain_mesh2d coordinate %s face all tessellation default LOCAL select_on invisible material orange selected_material default_selected render_shaded;\n' %(coordinates_str))

            writeRaw = True if (ALL and findNuclearProjections and not noRawData) or (not ALL) else False
            if not TRACT:
                if not noRawData:
                    w_out.write('gfx modify g_element /raw_data/ general clear;\n')
                for c,key in enumerate(nuclearGrouplist):
                    visibilitystr = ' invisible' if key in tract_namelist and TRACT else ''
                    # visibilitystr = ' invisible'
                    if c>0:
                        currentName = key[2:]
                        prevName = nuclearGrouplist[c-1][2:]
                        if currentName != prevName:
                            currentCol = cols[c%numcols] # otherwise, do not overwrite.
                    else:
                        currentCol = cols[c%numcols]
                    if midbrain_test and ALL:
                        w_out.write('gfx modify g_element "/" points domain_nodes subgroup "nuclear group %s" coordinate %s tessellation default_points LOCAL glyph sphere size "1*1*1" offset 0,0,0 font default scale_factors "1*1*1" select_on%s material %s selected_material default_selected render_shaded;\n' % (key, coordinates_str, visibilitystr, currentCol))
                        w_out.write('gfx modify g_element "/" points domain_nodes subgroup "nuclear group %s" coordinate %s tessellation default_points LOCAL glyph point size "1*1*1" offset 0,0,0 font default label brainstem_region_name label_offset 0,0,0 select_on material default selected_material default_selected render_shaded;\n' %(key,coordinates_str))
                        if key[:2] == 'R ':    print('midbrain_test: ',key[2:], 'z: %2.2f'%midbrainNucleiZVal[key])
                    if ALL and findNuclearProjections:
                        w_out.write('gfx modify g_element "/" points domain_nodes subgroup "%s_%s" coordinate %s tessellation default_points LOCAL glyph sphere size "%0.3f*%0.3f*%0.3f" offset 0,0,0 font default orientation deformed_orientation_scale scale_factors "1*1*1" select_on%s material %s selected_material default_selected render_shaded;\n' % (xiGroupRootName, key, coordStr, rs,rs,rs,visibilitystr, currentCol))
                    elif not ALL:
                        w_out.write('gfx modify g_element "/" points domain%s subgroup "xiGroups_%s" coordinate %s tessellation default_points LOCAL glyph sphere size "%0.3f*%0.3f*%0.3f" offset 0,0,0 font default orientation orientation_scale scale_factors "1*1*1" select_on%s material %s selected_material default_selected render_shaded;\n' % (domainStr, key, coordStr, rs,rs,rs,visibilitystr, currentCol))
                    if not noRawData and writeRaw:
                        w_out.write('gfx modify g_element /raw_data/ points domain_nodes subgroup "group %s" coordinate data_coordinates tessellation default_points LOCAL glyph diamond size "0.4*0.4*0.4" offset 0,0,0 font default select_on%s material %s selected_material default_selected render_shaded;\n' %(key, visibilitystr, currentCol))

            if writeBadFitRegions or BRN:# or TRACT:
                w_out.write('gfx modify g_element "/" points domain_nodes%s coordinate %s tessellation default_points LOCAL glyph none size "1*1*1" offset 0,0,0 font default label brainstem_region_name label_offset 0.5,0.5,0 select_on%s material default selected_material default_selected render_shaded;\n\n' %(subgroup_name, coordinates_str, visibilitystr))

            if include_cx and BRN:
                w_out.write('gfx define field ds1 node_value fe_field %s d/ds1\n' %coordinates_str)
                w_out.write('gfx modify g_element "/" lines domain_mesh1d subgroup intra-connections coordinate %s face all tessellation default LOCAL line line_base_size 0 select_on invisible material gold selected_material default_selected render_shaded;\n' %coordinates_str)

            if not BRN:
                w_out.write('gfx modify g_element "/" lines domain_mesh1d subgroup "nerve tracts" coordinate %s face all tessellation default LOCAL line_width 4 line line_base_size 0 select_on invisible material gold selected_material default_selected render_shaded;\n' %coordinates_str)
                w_out.write('gfx modify g_element "/" points domain_nodes subgroup "tract group" coordinate %s tessellation default_points LOCAL glyph none size "0.7*0.7*0.7" offset 0,0,0 font default label brainstem_region_name label_offset 0,0,0 select_on invisible material gold selected_material default_selected render_shaded;\n\n' %coordinates_str)

            visibilitystr = ' invisible' if BRN else ''
            if TRACT:# or BRN:
                for ic, cn in enumerate(existingCranialRegions):
                    for i in range(3):
                        w_out.write('gfx define field %s/d%d node_value fe_field %s d/ds%d\n' % (cn, i + 1, coordinates_str, i + 1))
                    w_out.write('gfx define field %s/orientation_scale composite d1 d2 d3\n\n' %cn)
                    w_out.write('gfx modify g_element "/%s/" general clear;\n' %cn)
                    sides = ['L ', 'R '] if cn not in midlineGroups else ''
                    for sidestr in sides:
                        w_out.write('gfx modify g_element "/%s/" points domain_nodes subgroup "%spoints" coordinate %s tessellation default_points LOCAL glyph point size "1*1*1" offset 0,0,0 font default label cranialnerve_object_name label_offset 0,0,0 select_on%s material %s selected_material default_selected render_shaded;\n' %(cn, sidestr, coordinates_str, visibilitystr, cols[ic%numcols]))
                        w_out.write('gfx modify g_element "/%s/" lines domain_mesh1d subgroup "%snerves" coordinate %s face all tessellation default LOCAL line_width 4 line line_base_size 0 select_on%s material %s selected_material default_selected render_shaded;\n' %(cn, sidestr, coordinates_str, visibilitystr, cols[ic%numcols]))
                    for nucleus in cranial_nerve_nuclei_list[cn.split(' ')[0]]:
                        w_out.write('gfx modify g_element "/%s/" points domain_nodes subgroup "nuclear group %s" coordinate %s tessellation default_points LOCAL glyph sphere size "%0.3f*%0.3f*%0.3f" offset 0,0,0 font default orientation orientation_scale scale_factors "0.2*0.2*0.2" select_on%s material %s selected_material default_selected render_shaded;\n' %(cn, nucleus, coordinates_str, rs,rs,rs,visibilitystr, cols[ic%numcols]))
                    w_out.write('gfx modify g_element "/%s/" points domain_nodes subgroup "external organ group" coordinate %s tessellation default_points LOCAL glyph point size "1*1*1" offset 0,0,0 font default label external_organ_name label_offset 0,0,0 select_on material %s selected_material default_selected render_shaded;\n' %(cn, coordinates_str, cols[ic%numcols]))
                    for im, mode in enumerate(modes):
                        w_out.write('gfx modify g_element "/%s/" lines domain_mesh1d subgroup %s-connections coordinate %s face all tessellation default LOCAL line_width 1 line line_base_size 0 select_on material %s selected_material default_selected render_shaded;\n' % (cn, mode, coordinates_str, cols[ic%numcols]))

            w_out.write("\ngfx create window\n")
            w_out.write("gfx edit scene\n")


# if True:
#     p = pstats.Stats('outProfile.txt')
#     p.strip_dirs().sort_stats(-1).print_stats()

toc = time.perf_counter()
print('ELAPSED TIME: ', toc-tic, ' s')