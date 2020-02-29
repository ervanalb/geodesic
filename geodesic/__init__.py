#!/usr/bin/env python3

import numpy as np
import itertools
from scipy.spatial import ConvexHull

phi = 0.5 * (1 + np.sqrt(5))
EPSILON = 1e-6

icosahedron_vertices = np.array([
    [0, 1, phi],
    [0, 1, -phi],
    [0, -1, phi],
    [0, -1, -phi],
    [1, phi, 0],
    [1, -phi, 0],
    [-1, phi, 0],
    [-1, -phi, 0],
    [phi, 0, 1],
    [phi, 0, -1],
    [-phi, 0, 1],
    [-phi, 0, -1],
]) / np.linalg.norm([1, phi])

tetrahedron_vertices = np.array([
    [1, 0, -np.sqrt(0.5)],
    [-1, 0, -np.sqrt(0.5)],
    [0, 1, np.sqrt(0.5)],
    [0, -1, np.sqrt(0.5)],
]) / np.linalg.norm([1, np.sqrt(0.5)])

octahedron_vertices = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
]) / np.linalg.norm([1, 1])

def field_z(pt):
    return np.array([1, 0, 0])

def field_radial(pt):
    return np.array(pt)

def field_from_vertices(vertices):
    def _field(pt):
        vectors = pt - vertices
        mags = np.maximum(np.linalg.norm(vectors, axis=1), 1e-5)
        return np.sum(vectors / mags[:,None] ** 2, axis=0)
    return _field

def field_from_faces(faces, vertices):
    face_centers = np.array([(vertices[i1] + vertices[i2] + vertices[i3]) / 3 for i1, i2, i3 in faces])

    def _field(pt):
        vectors = pt - face_centers
        mags = np.maximum(np.linalg.norm(vectors, axis=1), 1e-5)
        directions = np.cross(face_centers, pt)
        directions = directions / np.maximum(np.linalg.norm(directions, axis=1)[:,None], 1e-5)
        return np.sum(directions / mags[:,None], axis=0)
    return _field

def field_from_polyhedron(faces, vertices, curl_factor=1e-3):
    ffv = field_from_vertices(vertices)
    fff = field_from_faces(faces, vertices)
    return lambda pt: ffv(pt) + curl_factor * fff(pt)

def edges_from_faces(faces):
    edges = set()
    for f in faces:
        edges.add(frozenset((f[0], f[1])))
        edges.add(frozenset((f[0], f[2])))
        edges.add(frozenset((f[1], f[2])))

    return [list(e) for e in edges]

#def faces_from_points(points):
#    faces = []
#    for (i, j, k) in itertools.combinations(range(len(points)), 3):
#        o = points[i]
#        normal = np.cross(points[j] - o, points[k] - o)
#        sides = np.dot(points - o, normal)
#        if np.all(sides < EPSILON) or np.all(sides > -EPSILON):
#            faces.append([i, j, k])
#
#    return faces

def faces_from_points(points):
    return ConvexHull(points).simplices

def orient_edges(edges, points, field=field_z):
    """ Flips edges so that they align with the given vector field
    """
    def flip_edge(e):
        pt1, pt2 = [points[i] for i in e]
        midpoint = 0.5 * (pt1 + pt2)
        direction = np.dot(pt2 - pt1, field(midpoint))
        return direction < 0

    return [e[::-1] if flip_edge(e) else e for e in edges]

def orient_faces(faces, points, field=field_radial):
    """ Flips triangles so that they are as close as possible to isosceles in the ABA representation,
    and wound so that their normal aligns with the given vector field.
    """
    def sort_triangle(f):
        (a, b, c) = f
        vec1 = points[b] - points[a]
        vec2 = points[c] - points[b]
        centroid = (points[a] + points[b] + points[c]) / 3
        flip_winding = np.dot(np.cross(vec1, vec2), field(centroid)) < 0
        triangle = (c, b, a) if flip_winding else (a, b, c)
        # The middle point is the one that is abnormally close or abnormally far from the centroid
        distance_to_centroid = np.array([np.linalg.norm(points[i] - centroid) for i in triangle])
        middle_point = np.argmax(np.abs(distance_to_centroid - np.mean(distance_to_centroid)))
        triangle = (triangle * 3)[middle_point + 2:middle_point + 5]
        return list(triangle)

    return [sort_triangle(f) for f in faces]

def subdivide_triangle(pt1, pt2, pt3, v):
    a = (pt2 - pt1) / v
    b = (pt3 - pt1) / v
    return [pt1 + a * i + b * j for i in range(v + 1) for j in range(v + 1 - i)]

def deduplicate_points(points):
    new_points = np.empty(shape=(0, 3))
    for point in points:
        if not np.any(np.linalg.norm(new_points - point, axis=1) < EPSILON):
            new_points = np.vstack((new_points, point))
    return new_points

def subdivide_faces(faces, points, v):
    new_points = [pt for (i1, i2, i3) in faces for pt in subdivide_triangle(points[i1], points[i2], points[i3], v)]
    return deduplicate_points(new_points)

def project_points_to_sphere(points):
    return points / np.linalg.norm(points, axis=1)[:, None]

def matrix_for_vertex(point, field=field_z):
    z = point
    x = field(point)

    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    if np.linalg.norm(y) < EPSILON:
        x = np.array([1, 0, 0])
        y = np.cross(z, x)
        if np.linalg.norm(y) < EPSILON:
            x = np.array([0, 1, 0])
            y = np.cross(z, x)
            assert np.linalg.norm(y) >= EPSILON
    y = y / np.linalg.norm(y)
    x = np.cross(z, y)

    result = np.eye(4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    result[0:3, 3] = point
    return result

def matrix_for_edge(v1, v2):
    translation = 0.5 * (v1 + v2)
    x = v2 - v1

    x = x / np.linalg.norm(x)
    y = np.cross(translation, x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)

    result = np.eye(4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    result[0:3, 3] = translation
    return result

def matrix_for_face(v1, v2, v3):
    translation = (v1 + v2 + v3) / 3
    y = v2 - 0.5 * (v1 + v3)
    x = v3 - v1

    x = x / np.linalg.norm(x)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)

    result = np.eye(4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    result[0:3, 3] = translation
    return result

def vertex_matrices(points, field=field_z):
    return [matrix_for_vertex(point, field).tolist() for point in points]

def edge_matrices(edges, points):
    return [matrix_for_edge(points[i1], points[i2]).tolist() for i1, i2 in edges]

def face_matrices(faces, points):
    return [matrix_for_face(points[i1], points[i2], points[i3]).tolist() for i1, i2, i3 in faces]

def edge_lengths(edges, points):
    """ Returns a list parallel to edges where each entry is the length of that edge
    """
    return np.array([np.linalg.norm(points[i2] - points[i1]) for i1, i2 in edges])

def vertex_edges(edges):
    """ Returns a list parallel to vertices where each entry is a list of edge indices
    for edges incident on that point.
    """
    n_points = max(max(edges, key=max)) + 1
    result = [set() for _ in range(n_points)]
    for (ei, (vi1, vi2)) in enumerate(edges):
        result[vi1].add(ei)
        result[vi2].add(ei)
    return [list(edges) for edges in result]

def vertex_faces(faces):
    """ Returns a list parallel to vertices where each entry is a list of face indices
    for faces containing that point.
    """
    n_points = max(max(faces, key=max)) + 1
    result = [set() for _ in range(n_points)]
    for (ei, (vi1, vi2, vi3)) in enumerate(faces):
        result[vi1].add(ei)
        result[vi2].add(ei)
        result[vi3].add(ei)
    return [list(faces) for faces in result]

def face_triangles_2d(faces, points, face_matrices):
    """ Returns the three points of a face triangle in the coordinate frame of the face transform.
    These points correspond to the face when drawn in the XY plane and transformed by the face matrix.
    """
    def _tri(face, matrix):
        tri_pts = points[face]
        tri_pts = np.vstack((tri_pts.T, np.ones(3)))
        tri_pts = np.dot(np.linalg.inv(matrix), tri_pts)
        tri_pts = tri_pts[0:2,:].T
        return tri_pts
    return [_tri(face, matrix) for face, matrix in zip(faces, face_matrices)]

def sphere(v=2, base=None):
    """ Returns the vertices, edges, and faces of a geodesic sphere.
    """
    vs = base if base is not None else icosahedron_vertices
    fs = faces_from_points(vs)
    field = field_from_polyhedron(fs, vs)

    vs = subdivide_faces(fs, vs, v=v)
    vs = project_points_to_sphere(vs)
    fs = faces_from_points(vs)
    fs = orient_faces(fs, vs)
    es = edges_from_faces(fs)
    es = orient_edges(es, vs, field=field)

    return (vs, es, fs)

def main():
    import solid
    import solid.utils

    (vs, es, fs) = geodesic_sphere(v=8)

    #scale_factor = 1. / np.amax(np.abs(vs), axis=1)
    #new_vs = vs * scale_factor[:, None]
    #vs = 0.5 * vs + 0.5 * new_vs

    vms = vertex_matrices(vs)
    ems = edge_matrices(es, vs)
    fms = face_matrices(fs, vs)

    els = edge_lengths(es, vs)
    ves = vertex_edges(es)
    vfs = vertex_faces(fs)

    fts = face_triangles_2d(fs, vs, fms)

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

if __name__ == "__main__":
    main()
