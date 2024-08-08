"""
This script import geometry as 'step' file and performs the operation similar to gui. One face of the airfoil is fixed and the concentrated forces from external file are applied to the nodes of the airfoil surface.
"""
# Do not delete the following import lines
from abaqus import*
from abaqusConstants import *
backwardCompatibility.setValues(includeDeprecated=True, reportDeprecated=False)
import __main__

# ---------- imports ---------- #
import part
import step
import material
import section
import assembly
import step
import mesh
import load
import regionToolset
import numpy as np
# import matplotlib.pyplot as plt
# ----------------------------- #

#------------------- Import part geometry --------------------------------- #
def import_part(file_path, part_name):
  step = mdb.openStep(file_path, scaleFromFile=OFF)
  mdb.models['Model-1'].PartFromGeometryFile(name=part_name, geometryFile=step, combine=True, dimensionality=THREE_D, type=DEFORMABLE_BODY)
  p = mdb.models['Model-1'].parts[part_name]
  # session.viewports['Viewport: 2'].setValues(displayedObject=p)

# ------------------- Define material ------------------------------------ #
def material(material_name, youngs_modulus, poisson_ratio):
  mdb.models['Model-1'].Material(name=material_name)
  mdb.models['Model-1'].materials[material_name].Elastic(table=((youngs_modulus, poisson_ratio), ))

# ------------------- Create section ------------------------------------ #
def create_section(section_name, material_name, part_name):
  mdb.models['Model-1'].HomogeneousSolidSection(name=section_name, material=material_name, thickness=None)
  p = mdb.models['Model-1'].parts[part_name]
  region = (p.cells,)
  p = mdb.models['Model-1'].parts[part_name]
  p.SectionAssignment(region=region, sectionName=section_name, offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

# ------------------- Create Assembly------------------------------------ #
def assembly(part_name, instance_name):
  a = mdb.models['Model-1'].rootAssembly
  a.DatumCsysByDefault(CARTESIAN)
  p = mdb.models['Model-1'].parts[part_name]
  a.Instance(name=instance_name, part=p, dependent=ON)

# ------------------- Create Step------------------------------------ #
def create_step(step_name):
  a = mdb.models['Model-1'].rootAssembly
  mdb.models['Model-1'].StaticStep(name=step_name, previous='Initial')

# ------------------- Create Mesh ------------------------------------ #
def mesh(part_name, edge_coordinates, set_name, fine_mesh_size, global_mesh_size):
  p = mdb.models['Model-1'].parts[part_name]
  e = p.edges
  ref_set = ()
  for num in range(6):
    myEdges = e.findAt(edge_coordinates[num],)
    ref_set = ref_set + (e[myEdges.index:myEdges.index+1],)

  edgeSet = p.Set(edges=ref_set, name=set_name)
  pickedEdges = tuple(edgeSet.edges)
  p.seedEdgeBySize(edges=pickedEdges, size=fine_mesh_size, deviationFactor=0.1, constraint=FINER)
  p.seedPart(size=global_mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
  p.generateMesh()


def create_surfaceTraction(step_name):
  '''
  a = mdb.models['Model-1'].rootAssembly
  el = a.instances['my_instance'].faces
  face_coordinates = ((250.E-03,-29.71E-03,25.E-03),(250.E-03,-29.71E-03,525.E-03),(600.202E-03,-22.812E-03,50.E-03),(950.E-03,-4.03E-03,525.E-03),(975.E-03,-2.33E-03,50.E-03),(975.E-03,-2.33E-03,1.),)
  face_set = ()
  for num2 in range(6):
    myFaces = el.findAt(face_coordinates[num2],)
    face_set = face_set + (el[myFaces.index:myFaces.index+1],)
    print("face set is ", face_set)
  a.Surface(side1Faces=face_set, name='Surf-1')

  surface = el.faces['Surf-1']
  elementSet = surface.elements


  '''
  # elementSet = ()

  # pressure_values = np.linspace(2, 5, len(elementSet))  # pressure values

  from numpy import genfromtxt

  # Load the CSV file
  data_pressure = genfromtxt('fine_pressure_abaqus.txt')
  data_s5 = genfromtxt('element_s5.txt')
  data_s3 = genfromtxt('element_s3.txt')
  data_s4 = genfromtxt('element_s4.txt')
  data_s6 = genfromtxt('element_s6.txt')

  # Access the pressure values as an array
  pressure_values = data_pressure  # Assuming pressure values are in the first column

  element_s5 = data_s5
  element_s5 = tuple(int(x) for x in element_s5)

  element_s3 = data_s3
  element_s3 = tuple(int(x) for x in element_s3)

  element_s4 = data_s4
  element_s4 = tuple(int(x) for x in element_s4)

  element_s6 = data_s6
  element_s6 = tuple(int(x) for x in element_s6)
  # print(pressure_values)




  a = mdb.models['Model-1'].rootAssembly
  f2 = a.instances['my_instance'].elements




  # create a loop to apply pressure on each face
  count = 0
  count_e5 = 0
  count_e3 = 0
  count_e4 = 0
  count_e6 = 0
  for pressure_value in zip(pressure_values):
    a = mdb.models['Model-1'].rootAssembly
    a.regenerate()
    f2 = a.instances['my_instance'].elements

    if count < len(element_s5):
      elementSequence = f2.sequenceFromLabels((element_s5[count_e5],))  # get the sequence to create the surface
      region = a.Surface(face5Elements=elementSequence, name='Surface'+str(count))
      mdb.models['Model-1'].SurfaceTraction(name='Load'+str(count), createStepName=step_name, region=region, magnitude=pressure_values[count], directionVector=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), distributionType=UNIFORM, field='', localCsys=None, traction=GENERAL, resultant=ON)

      count_e5 = count_e5+1

    if len(element_s5)<= count < (len(element_s5)+len(element_s3)):
      elementSequence = f2.sequenceFromLabels((element_s3[count_e3],))  # get the sequence to create the surface
      region = a.Surface(face3Elements=elementSequence, name='Surface'+str(count))
      mdb.models['Model-1'].SurfaceTraction(name='Load'+str(count), createStepName=step_name, region=region, magnitude=pressure_values[count], directionVector=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), distributionType=UNIFORM, field='', localCsys=None, traction=GENERAL, resultant=ON)

      count_e3 = count_e3+1

    if (len(element_s5)+len(element_s3))<= count < (len(element_s5)+len(element_s3)+len(element_s4)):
      elementSequence = f2.sequenceFromLabels((element_s4[count_e4],))  # get the sequence to create the surface
      region = a.Surface(face4Elements=elementSequence, name='Surface'+str(count))
      mdb.models['Model-1'].SurfaceTraction(name='Load'+str(count), createStepName=step_name, region=region, magnitude=pressure_values[count], directionVector=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), distributionType=UNIFORM, field='', localCsys=None, traction=GENERAL, resultant=ON)

      count_e4 = count_e4+1

    if (len(element_s5)+len(element_s3)+len(element_s4))<= count < (len(element_s5)+len(element_s3)+len(element_s4)+len(element_s6)):
      elementSequence = f2.sequenceFromLabels((element_s6[count_e6],))  # get the sequence to create the surface
      region = a.Surface(face6Elements=elementSequence, name='Surface'+str(count))
      mdb.models['Model-1'].SurfaceTraction(name='Load'+str(count), createStepName=step_name, region=region, magnitude=pressure_values[count], directionVector=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), distributionType=UNIFORM, field='', localCsys=None, traction=GENERAL, resultant=ON)

      count_e6 = count_e6+1

    count = count+1




# -------------- required parameters ------------------------------- #
refineEdges = ((61.344E-03,12.564E-03,1.),(61.344E-03,-12.564E-03,1.),(300.101E-03,11.406E-03,1.),(300.101E-03,-11.406E-03,1.),(487.5E-03,1.165E-03,1.),(487.5E-03,-1.165E-03,1.),)

# -------------- calling the required functions -------------------- #
import_part(file_path='/mnt/home_ilrw/ruki065d/ThesisData/abaqusFiles/aero_airfoil/NACA0006_airfoil_train4.step', part_name='NACA0006_airfoil_train')
material(material_name='my_material', youngs_modulus=1000000, poisson_ratio=0.3)
create_section(section_name='my_section', material_name='my_material', part_name='NACA0006_airfoil_train')
assembly(part_name='NACA0006_airfoil_train', instance_name='my_instance')
create_step(step_name='my_pressure')
mesh(part_name='NACA0006_airfoil_train', edge_coordinates=refineEdges, set_name='edge_ref_set', fine_mesh_size=0.0042, global_mesh_size=0.009)
create_surfaceTraction(step_name='my_pressure')



