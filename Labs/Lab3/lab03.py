import os
import geopandas as gpd

# Define the base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the GeoJSON file using the absolute path 
gdf = gpd.read_file(r"C:\Users\madan.sapkota\OneDrive - Texas A&M AgriLife\Madan PhD TAMU\Graduate Courses\Fall 2024\GIS Programming 676\Geog676_MS\Labs\Lab3\data.geojson")


class CensusTract:
    #Represents a census tract, a geographic area with population data.
    
    def __init__(self, geoid, population, geometry):
        """
        Initialize the CensusTract with geoid (identifier), population, and geometry (spatial boundary).
        - geoid: unique identifier for the census tract
        - population: total population in the tract
        - geometry: spatial boundaries of the census tract (a Polygon)
        """
        self.geoid = geoid
        self.population = population
        self.geometry = geometry
    
    def calculate_population_density(self):
        """
        Calculate the population density (people per square kilometer).
        - Population density is calculated by dividing population by the area of the tract.
        - The area is initially in square meters and needs to be converted to square kilometers.
        """
        area = self.geometry.area  # Calculate the area in square meters
        if area > 0:
            population_density = self.population / (area / 10**6)  # Convert square meters to square kilometers
        else:
            population_density = 0  # If area is zero, set population density to 0
        return population_density

if __name__ == "__main__":
    # Load data directly from the specified GeoJSON file path again to ensure fresh data for main script execution
    gdf = gpd.read_file(r"C:\Users\madan.sapkota\OneDrive - Texas A&M AgriLife\Madan PhD TAMU\Graduate Courses\Fall 2024\GIS Programming 676\Geog676_MS\Labs\Lab3\data.geojson")

    # Preview the first 5 rows of the data to understand its structure
    print(gdf.head())        # Display the first 5 rows of the GeoDataFrame
    print(gdf.columns)       # Show the column names to see available data
    print(gdf.shape)         # Show the shape of the GeoDataFrame (number of rows, columns)
    print(gdf.dtypes)        # Display the data types of each column for debugging purposes

    # Check the current Coordinate Reference System (CRS) used by the GeoDataFrame
    print("Original CRS:", gdf.crs)

    # Reproject the GeoDataFrame to a more appropriate projected coordinate system (e.g., UTM Zone 15N)
  
    gdf = gdf.to_crs(epsg=26915)  # EPSG: 26915 corresponds to UTM Zone 15N


def calculate_density(row):
    """
    Calculate population density for a given row (census tract).
    - Each row represents a census tract with population and geometry.
    - Instantiate the CensusTract class for each row and call the `calculate_population_density` method.
    """
    census_tract = CensusTract(geoid=row['GeoId'], population=row['Pop'], geometry=row['geometry'])
    return census_tract.calculate_population_density()

# Apply the population density calculation for each row of the GeoDataFrame
# The `apply` function applies the `calculate_density` function to each row (axis=1 means row-wise operation)
gdf['Pop_den_new'] = gdf.apply(calculate_density, axis=1)

# Preview the updated GeoDataFrame to check the new 'Pop_den_new' column
print(gdf.head())

















































