### Feature extraction
[x] surface area
[x] compactness (with respect to a sphere) IS STILL WRONG!!! Must be at most 1!
[x] axis-aligned bounding-box volume
[x] diameter
[x] eccentricity
[x] A3: angle between 3 random vertices
[x] D1: distance between barycenter and random vertex
[x] D2: distance between 2 random vertices
[x] D3: square root of area of triangle given by 3 random vertices
[x] D4: cube root of volume of tetrahedron formed by 4 random vertices

### 
[] Mention in paper: Rectangularity

### Queriying
[] Explain how the query occurs procedurally. With algorithm.
[] For the full feature variant use Mahabolis distance d(x,y) = sqrt((x-y).T * C^-1 * (x-y)) where C is the data covariance matrix
[] Talk about feature weighting and distance weighting 
    -> For distance weighting $d(A,B) = w_1*d(a_{1..5},b_{1..5}) + w_2*d(a_{6..n, 6..n) + ... $
    -> For feature weighting standardization 
    -> Distance weighting solved with grid search. Why: More flexible way of finding optimal weight combinations. Standard deviation method assumes that all features have same importance with respect to the database. That is not necessarily the case. A database of planes might not have much use for the skeleton endpoint feature for instance. 

### Scalability
[] TSNE: Use our pairwise distance function

### Personal tasks
[x] Parallelize normalisation operation 
[] For all vector query use the right weighting
[] Histograms for before and after normalisation
[] Implement rest of the features on the slide (Convex hull and non convex hull, Average curvature, Reflective Symmetry, Morphological features)


### Clean Up tasks (For the end)
[] Read all the technical tips again
[] Streamline the formulas in the paper
[] Proof read the paper
[] Rerun the full process
[] Add all package references
[] Streamline cells/faces and point/vertex usage

