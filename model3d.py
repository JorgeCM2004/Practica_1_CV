"""
author: José M. Buenaposada (josemiguel.buenaposada@urjc.es)
date: 2025/03
"""

import numpy as np
import cv2

class Model3D:
    def __init__( self ):
        self.vertices = None
        self.faces = None
        self.face_centroids = None
        self.color_map = None

    def load_from_obj(self, file_path):
        """
        Adapted from: https://medium.com/@harunijaz/the-code-above-is-a-python-function-that-reads-and-loads-data-from-a-obj-e6f6e5c3dfb9

        :param file_path: full path of the .obj file to load
        """
        try:
            vertices = []
            faces = []
            with open(file_path) as f:
               for line in f:
                   if line[0] == "v":
                       vertex = list(map(float, line[2:].strip().split()))
                       vertices.append(vertex)
                   elif line[0] == "f":
                       face = list(map(int, line[2:].strip().split()))
                       faces.append(face)

            self.vertices = np.array(vertices)
            self.faces = np.array(faces)

            # Compute triangle centroids (for z-buffer plot)
            self.face_centroids = []
            for i in range(self.faces.shape[0]):
                p0 = self.faces[i, 0] - 1
                p1 = self.faces[i, 1] - 1
                p2 = self.faces[i, 2] - 1
                vertices = np.vstack((self.vertices[p0, :],
                                      self.vertices[p1, :],
                                      self.vertices[p2, :]))
                self.face_centroids.append(np.mean(vertices, axis=0))
            self.face_centroids = np.array(self.face_centroids)

        except FileNotFoundError:
            print(f"{file_path} not found.")
        except:
            print("An error occurred while loading the shape.")

    def translate(self, t):
        assert t.shape == (1, 3)

        if not self.vertices is None:
            self.vertices += t
            self.face_centroids += t

    def scale(self, scale):
        assert isinstance(scale, float)

        if not self.vertices is None:
            self.vertices *= scale
            self.face_centroids *= scale

    def plot_on_image(self, img, P):

        if self.color_map is None:
            self.color_map = np.int32(255 * np.random.rand(self.faces.shape[0], 3))

        vertices3D = self.vertices.copy()
        vertices3D = np.hstack((vertices3D, np.ones((vertices3D.shape[0], 1))))
        vertices2D = P @ vertices3D.T
        vertices2D /= vertices2D[2, : ]
        vertices2D = np.int32(np.round(vertices2D))

        # Compute distance of each triangle to the camera
        distances = []
        centroids = np.hstack((self.face_centroids, np.ones((self.face_centroids.shape[0],1)))).T
        for i in range(self.faces.shape[0]):
            triangle_depth = P[2, :] @ centroids[:, i]
            triangle_depth /= np.linalg.norm(P[2,0:3])
            triangle_depth *= np.sign(np.linalg.det(P[0:3,0:3]))
            distances.append(triangle_depth)

        # Plot in projected into image triangles first further away from
        # camera ones (z-buffer idea).
        distances = np.array(distances)
        dist_sorted_indices = distances.argsort()[::-1]
        for i in dist_sorted_indices:
            p0 = self.faces[i, 0]-1
            p1 = self.faces[i, 1]-1
            p2 = self.faces[i, 2]-1
            pt0 = vertices2D[0:2, p0]
            pt1 = vertices2D[0:2, p1]
            pt2 = vertices2D[0:2, p2]
            color = (int(self.color_map[i,0]),
                     int(self.color_map[i,1]),
                     int(self.color_map[i,2]))
            pts = np.vstack((pt0, pt1, pt2))
            cv2.fillPoly(img, [pts], color)
            
            
