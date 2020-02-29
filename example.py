import solid
import solid.utils
import geodesic as geo

(vs, es, fs) = geo.sphere(v=8)

#scale_factor = 1. / np.amax(np.abs(vs), axis=1)
#new_vs = vs * scale_factor[:, None]
#vs = 0.5 * vs + 0.5 * new_vs

vms = geo.vertex_matrices(vs)
ems = geo.edge_matrices(es, vs)
fms = geo.face_matrices(fs, vs)

els = geo.edge_lengths(es, vs)
ves = geo.vertex_edges(es)
vfs = geo.vertex_faces(fs)

fts = geo.face_triangles_2d(fs, vs, fms)

shape = solid.union()
#for e, em, el in zip(es, ems, els):
#    edge_shape = solid.utils.rot_z_to_right(solid.cylinder(r1=0.05, r2=0.01, h=el * 0.9, center=True))
#    if any(len(ves[i]) == 5 for i in e):
#        edge_shape = solid.color("red")(edge_shape)
#    shape.add(solid.multmatrix(em)(edge_shape))

for f, fm, ft in zip(fs, fms, fts):
    face_shape = solid.linear_extrude(0.01)(solid.polygon(ft * 0.8))
    if any(len(vfs[i]) == 5 for i in f):
        face_shape = solid.color("red")(face_shape)
    shape.add(solid.multmatrix(fm)(face_shape))

solid.scad_render_to_file(shape, 'test.scad')

#edge_matrix = [matrix_for_edge(i, vertex_translations, edge_vertex_indices).tolist() for i in edges]
#face_matrix = [matrix_for_face(i, vertex_translations, edge_vertex_indices, face_edge_indices).tolist() for i in faces]
#print("for (m = vertex_matrix) { multmatrix(m) cylinder(r=0.1, h=0.2, center=true);}")
#print("for (m = vertex_matrix) { multmatrix(m) sphere(r=0.1, center=true);}")
#print("for (m = vertex_matrix) { multmatrix(m) linear_extrude(0.1) text(\"A\", 0.3);}")
#print("for (m = edge_matrix) { multmatrix(m) rotate(90, [0, 1, 0]) cylinder(r1=0.05, r2=0.01, h=0.4, center=true);}")
#print("for (m = face_matrix) { multmatrix(m) rotate(90, [1, 0, 0]) cylinder(r1=0, r2=0.1, h=0.2, center=true);}")
#print()
#print("$fn = 12;")
#print("vertex_matrix = " + str(vertex_matrix) + ";")
#print("edge_matrix = " + str(edge_matrix) + ";")
#print("face_matrix = " + str(face_matrix) + ";")
#print("vertex_translations = " + str(vertex_translations.tolist()) + ";")
#f = faces_from_points(icosahedron_vertices)
#p = subdivide_faces(f, icosahedron_vertices, v=3)
#print(p)
#e = edges_from_faces(f)

#print(e)
#print(orient_edges(e, icosahedron_vertices))
#print(orient_faces(f, icosahedron_vertices))
