PARKS_metadata = {'tags': ['activities', 'environment', 'green', 'health', 'nature', 'parks', 'walk', 'youth'], 'category': 'environment'}

IMPORTANTTREES_metadata = {'tags': ['heritage', 'preservation', 'tree'], 'category': 'recreation-and-culture', 'maintained by': ['parks division (public property trees)', 'planning (private property trees)']}

PARKSPECIMENTREES_metadata = {'tags': ['trees'], 'category': 'environment'}

PARKS_schema = {'facilityid': {'type': 'string', 'length': 20}, 'location': {'type': 'string', 'length': 254}, 'park_name': {'type': 'string', 'length': 50}, 'comments': {'type': 'string', 'length': 1500}, 'legacyid': {}, 'operating_location': {}}

# coded values left out
IMPORTANTTREES_schema = {'site_name': {'type': 'string', 'length': 50}, 'tree_species': {'type': 'string', 'alias': 'species', 'length': 30, 'coded values': []}, 'tree_status': {'type': 'string', 'alias': 'status', 'length': 15, 'coded values': []}, 'owner': {'type': 'string', 'length': 45, 'coded values': []}, 'latitude': {}, 'longitude': {}}

PARKSPECIMENTREES_schema = {'description': {'type': 'string', 'length': 100}, 'location': {'type': 'string', 'length': 254}, 'park': {'type': 'string', 'length': 50}, 'tree_genus': {'type': 'string', 'length': 50, 'coded values': []}, 'tree_species': {'type': 'string', 'length': 50, 'coded values': []}, 'tree_variety': {'type': 'string', 'length': 50, 'coded values': []}, 'tree_type': {'type': 'string', 'length': 12, 'coded values': []}, 'operating_location': {}}

PARKS_data = {'facilityid': [1001487039, 1001879098], 'location': ['19340 88 Ave', '15999 28 Ave'], 'park_name': ['Port Kells Park', 'Wills Brook Park'], 'comments': [None, '- Formerly 114M Greenbelt, 114H Neighbourhood Park, 114J Greenbelt'], 'legacyid': ['0701-000000537', None], 'operating_location': ['Port Kells Park, 19340 88 Ave', 'Wills Brook Park, 15999 28 Ave']}

IMPORTANTTREES_data = {'site_name': ['Port Kells Park - 19340 88 Ave', 'King George Blvd - 2251', 'Cloverdale Elementary School'], 'tree_species': ['Douglas Fir', 'English Oak', 'Horse Chestnut'], 'tree_status': ['Operating', 'Operating', 'Operating'], 'owner': ['City of Surrey', 'City of Surrey', 'School District #36'], 'latitude': [49.159741, 49.042877, 49.104314
], 'longitude': [-122.686525, -122.790481, -122.728366]}

PARKSPECIMENTREES_data = {'description': ['Pseudotsuga menziesii; Port Kells Park, 19340 88 Ave', 'Quercus robur, 2251 King George Blvd', "Gleditsia triacanthos 'Shademaster'; 13507 87B Ave"], 'location': ['In the park, tree ref # 45', 'Blvd, KGBlvd, 156 St to 24 Ave, W side, tree ref # 17', 'On the boulevard'], 'park': ['Port Kells Park', None, None], 'tree_genus': ['Pseudotsuga',
'Quercus', 'Gleditsia'], 'tree_species': ['menziesii', 'robur', 'triacanthos'], 'tree_variety': [None, None, 'Shademaster'], 'tree_type': ['Park', 'Boulevard', 'Boulevard'], 'operating_location': ['Port Kells Park, 19340 88 Ave', None, None]}


DATASOURCES = {'Parks': ('Parks', PARKS_metadata, PARKS_schema, PARKS_data),
               'ImportantTrees': ('ImportantTrees', IMPORTANTTREES_metadata, IMPORTANTTREES_schema, IMPORTANTTREES_data),
               'ParkSpecimenTrees': ('ParkSpecimenTrees', PARKSPECIMENTREES_metadata, PARKSPECIMENTREES_schema, PARKSPECIMENTREES_data)
               }

KNOWLEDGE_BASE = {'species':
                      {'mappings': {'ImportantTrees.tree_species':
                                        {'source': 'ImportantTrees',
                                        'attribute_name': 'tree_species',
                                        'match_score': 0,
                                        'sample_values': [],
                                        'data_type': None},
                                    'ParkSpecimenTrees.tree_species':
                                        {}
                                    },
                       'synonyms': {'tree_species': 2},
                       'is-parent-of': {},
                       'is-child-of': {'genus': []}
                       },
                  # similarly for the other concepts
                  'park':
                      {'mappings': {},
                       'synonyms': {},
                       'is-parent-of': {},
                       'is-child-of': {}
                       }
                  }

# for i in list(PARKS_schema.keys()):
#     print(i)
# print()
# for i in list(IMPORTANTTREES_schema.keys()):
#     print(i)
# print()
# for i in list(PARKSPECIMENTREES_schema.keys()):
#     print(i)
