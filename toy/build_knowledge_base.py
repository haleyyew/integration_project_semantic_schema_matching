import data

# these are the matchers to be implemented
name_matcher, name_matcher_with_thesaurus, name_matcher_with_metadata, attribute_type_matcher, bayesian_matcher, deep_neural_net = None, None, None, None, None, None

def perform_schema_matching(source_schema, target_schema):
    correspondences_with_score = {}
    # run schema matching algorithm
        # name matcher
        # name matcher with thesaurus
        ## name matcher with metadata, improved after each iteration; not useful
        # attribute type matcher

    matchers = {'name_matcher': name_matcher, 'name_matcher_with_thesaurus': name_matcher_with_thesaurus,  'attribute_type_matcher': attribute_type_matcher}

    # ----- for testing purposes -----
    # output after 1st iteration
    if target_schema[0] == 'Parks':
        correspondences_with_score = {
            'name_matcher': {
                ('parks', 'park_name'): 0.5,
                ('parks division (public property trees)', 'park_name'): 0.10
            },
            'name_matcher_with_thesaurus': {
                ('parks', 'park_name'): 0.5,
                ('parks', 'parks division (public property trees)'): 0.10
            },
            'attribute_type_matcher': {}
        }
    elif target_schema[0] == 'ImportantTrees':
        correspondences_with_score = {
            'name_matcher': {
                ('tree', 'tree_species'): 0.4,
                ('tree', 'tree_status'): 0.4
            },
            'name_matcher_with_thesaurus': {
                ('tree', 'tree_species'): 0.4,
                ('tree', 'tree_status'): 0.4
            },
            'attribute_type_matcher': {}
        }
    elif target_schema[0] == 'ParkSpecimenTrees':
        correspondences_with_score = {
            'name_matcher': {
                ('parks', 'park'): 0.8,
                ('parks division (public property trees)', 'park'): 0.2,
                ('tree', 'tree_genus'): 0.5,
                ('tree', 'tree_species'): 0.5,
                ('tree', 'tree_variety'): 0.5,
                ('tree', 'tree_type'): 0.5
            },
            'name_matcher_with_thesaurus': {
                ('parks', 'park'): 0.8,
                ('parks division (public property trees)', 'park'): 0.2,
                ('tree', 'tree_genus'): 0.5,
                ('tree', 'tree_species'): 0.5,
                ('tree', 'tree_variety'): 0.5,
                ('tree', 'tree_type'): 0.5
            },
            'attribute_type_matcher': {}
        }
    # --------------------------------

    return correspondences_with_score

def perform_entity_matching(source_dataset, target_dataset):
    correspondences_with_score = {}
    # run entity matching algorithm
        # bayesian matcher
        # deep neural network

    matchers = {'bayesian_matcher': bayesian_matcher, 'deep_neural_net': deep_neural_net}

    # ----- for testing purposes -----
    # output after 1st iteration
    if target_dataset[0] == 'Parks':
        correspondences_with_score = {
            'bayesian_matcher': {},
            'deep_neural_net': {}
        }
    elif target_dataset[0] == 'ImportantTrees':
        correspondences_with_score = {
            'bayesian_matcher': {},
            'deep_neural_net': {}
        }
    elif target_dataset[0] == 'ParkSpecimenTrees':
        correspondences_with_score = {
            'bayesian_matcher': {},
            'deep_neural_net': {}
        }
    # --------------------------------

    return correspondences_with_score

def build_concept(datasource, correspondences_with_score, score_threshold):
    set_of_concepts_with_score = {}
    # filter correspondences with high score
    for correspondence in correspondences_with_score:
        score = correspondences_with_score[correspondence]
        if score < score_threshold:
            del correspondences_with_score[correspondence]

    for correspondence in correspondences_with_score:
        concept = find_concept_in_dictionary(correspondence)

    # ----- for testing purposes -----
    # output after 1st iteration
    if datasource[0] == 'Parks':
        set_of_concepts_with_score = {
            ('park', 'park_name'): 0.2          # 's' is removed
        }
    elif datasource[0] == 'ImportantTrees':
        set_of_concepts_with_score = {
            ('species', 'tree_species'): 0.16,  # 'tree' changed to 'species'
            ('tree', 'tree_status'): 0.16
        }
    elif datasource[0] == 'ParkSpecimenTrees':
        set_of_concepts_with_score = {
            ('park', 'park'): 0.32,             # 'parks' changed to 'park'
            ('genus', 'tree_genus'): 0.2,       # 'tree' changed to 'genus'
            ('species', 'tree_species'): 0.2,   # 'tree' changed to 'species'
            ('tree', 'tree_variety'): 0.2,
            ('tree', 'tree_type'): 0.2
        }
    # --------------------------------

    return set_of_concepts_with_score

def find_concept_in_dictionary(correspondence):
    concept = None
    # lookup in WordNet
    # might look up the value of an attribute to figure out a better concept name
    return concept

def merge_concepts_into_knowledge_base(knowledge_base, set_of_concepts, datasource):
    # update knowledge base with the set of concepts

    # ----- for testing purposes -----
    if datasource is None:
        return
    # output after 1st iteration
    if datasource[0] == 'Parks':
        knowledge_base['park'] = {
            'mappings': {'Parks.park_name':
                                {'source': 'Parks',
                                'attribute_name': 'park_name',
                                'match_score': 0.2,
                                'sample_values': ['Port Kells Park'],    # find some sample values by matrix factorization
                                'data_type': {'type': 'string', 'length': 50}}
                         },
            'synonyms': {},
            'is-parent-of': {},
            'is-child-of': {}
            }
    elif datasource[0] == 'ImportantTrees':
        knowledge_base['species'] = {
            'mappings': {'ImportantTrees.tree_species':
                                {'source': 'ImportantTrees',
                                'attribute_name': 'tree_species',
                                'match_score': 0.16,
                                'sample_values': ['Douglas Fir'],
                                'data_type': {'type': 'string', 'length': 30}}
                         },
            'synonyms': {'tree_species': 1},
            'is-parent-of': {},
            'is-child-of': {}
            }
        # do the same procedure as above
        knowledge_base['tree'] = {}
    elif datasource[0] == 'ParkSpecimenTrees':
        knowledge_base['park']['mappings']['ParkSpecimenTrees.park'] = {
            'source': 'ParkSpecimenTrees',
            'attribute_name': 'park',
            'match_score': 0.32,
            'sample_values': ['Port Kells Park'],
            'data_type': {'type': 'string', 'length': 50}
        }
        # do the same procedure as above
        knowledge_base['species']['mappings']['ParkSpecimenTrees.tree_species'] = {}
        knowledge_base['genus'] = {}
        knowledge_base['tree'] = {}
        # note 'tree_type' is not merged into the knowledge base because there is already a 'tree' concept mapping for this source
            # if want to include it, need a probabilistic method!

    # --------------------------------

    return

def generate_new_concepts(knowledge_base, datasource):
    set_of_concepts = {}
    # perform schema summarization (need a new approach, existing method might not work due to value links)
    # perform dataset summarization (by matrix factorization)

    # ----- for testing purposes -----
    if datasource[0] == 'Parks':
        set_of_concepts = {'primary key', 'location'}   # these concepts were determined from schema and dataset summarization
    elif datasource[0] == 'ImportantTrees':
        set_of_concepts = {'site name', 'City of Surrey', 'School District'}
    elif datasource[0] == 'ParkSpecimenTrees':
        set_of_concepts = {'location', 'Boulevard'}
    # --------------------------------

    # then added the new concepts to the knowledge base

    return set_of_concepts

def hybrid_schema_matching(knowledge_base, datasource):
    correspondences_with_score = {}
    correspondences_with_score_schema = perform_schema_matching(datasource, knowledge_base)
    correspondences_with_score_entity = perform_entity_matching(datasource, knowledge_base)
    correspondences_with_score.update(correspondences_with_score_schema)
    correspondences_with_score.update(correspondences_with_score_entity)

    correspondences_with_score = compose_matching_score(datasource, correspondences_with_score)
    return correspondences_with_score

def compose_matching_score(datasource, correspondences_with_score):
    correspondences_with_score = {}
    # check whether correspondences agree with each other across all matchers
    # compute probabilities based on correspondences that don't agree
    # weights of matchers are set in advance

    # ----- for testing purposes -----
    weights = {'name_matcher': 0.2, 'name_matcher_with_thesaurus': 0.2,  'attribute_type_matcher': 0.2, 'bayesian_matcher': 0.2, 'deep_neural_net': 0.2}

    # output after 1st iteration
    if datasource[0] == 'Parks':
        correspondences_with_score = {
            # because the attribute_type_matcher, bayesian_matcher, deep_neural_net return no correspondences, they are ommited in the other computations
            ('parks', 'park_name'): 0.5 * weights['name_matcher'] + 0.5 * weights['name_matcher_with_thesaurus'] + 0 * weights['attribute_type_matcher'] + 0 * weights['bayesian_matcher'] + 0 * weights['deep_neural_net'],
            ('parks division (public property trees)', 'park_name'): 0.1 * weights['name_matcher'] + 0.1 * weights['name_matcher_with_thesaurus']
        }
    elif datasource[0] == 'ImportantTrees':
        correspondences_with_score = {
            ('tree', 'tree_species'): 0.4 * weights['name_matcher'] + 0.4 * weights['name_matcher_with_thesaurus'],
            ('tree', 'tree_status'): 0.4 * weights['name_matcher'] + 0.4 * weights['name_matcher_with_thesaurus']
        }
    elif datasource[0] == 'ParkSpecimenTrees':
        correspondences_with_score = {
            ('parks', 'park'): 0.8 * weights['name_matcher'] + 0.8 * weights['name_matcher_with_thesaurus'],
            ('parks division (public property trees)', 'park'): 0.2 * weights['name_matcher'] + 0.2 * weights['name_matcher_with_thesaurus'],
            ('tree', 'tree_genus'): 0.5 * weights['name_matcher'] + 0.5 * weights['name_matcher_with_thesaurus'],
            ('tree', 'tree_species'): 0.5 * weights['name_matcher'] + 0.5 * weights['name_matcher_with_thesaurus'],
            ('tree', 'tree_variety'): 0.5 * weights['name_matcher'] + 0.5 * weights['name_matcher_with_thesaurus'],
            ('tree', 'tree_type'): 0.5 * weights['name_matcher'] + 0.5 * weights['name_matcher_with_thesaurus']
        }
    # --------------------------------

    return correspondences_with_score

def collect_concepts_from_tags(datasources):
    set_of_concepts = {}
    # collect all the tags from datasets
    for datasource in datasources:
        print(datasources[datasource][0])
        metadata = datasources[datasource][1]

        for key in metadata:
            concepts = {}
            if isinstance(metadata[key], list):
                concepts = {i: None for i in metadata[key]}
            else:
                concepts = {metadata[key]: None}
            set_of_concepts.update(concepts.copy())

    # for i in list(set_of_concepts.keys()):
    #     print(i)

    return set_of_concepts

def check_dataset_covered_by_concepts(knowledge_base, datasource):
    coverage_percentage = 0
    coverage = {}
    # from entity matching output, report how many instances are covered by a concept
        # entity matcher should report the correspondences, i.e. which values in the data instance
        # find all values that don't appear in correspondences
    # from schema matching, report how many attributes are not in any correspondences

    # ----- for testing purposes -----
    weights = {'entity_matching': 0.5, 'schema_matching': 0.5}

    # output after 1st iteration
    if datasource[0] == 'Parks':
        coverage['facilityid'] = 0
        coverage['location'] = 0
        coverage['park_name'] = 1 * weights['schema_matching'] + 0 * weights['entity_matching']
        coverage['comments'] = 0
        coverage['legacyid'] = 0
        coverage['operating_location'] = 0
        coverage_percentage = 0.5/6
    elif datasource[0] == 'ImportantTrees':
        coverage_percentage = (2 * 0.5)/6
    elif datasource[0] == 'ParkSpecimenTrees':
        coverage_percentage = (4 * 0.5)/8
    # --------------------------------

    return coverage_percentage

def update_datasource_with_metadata(knowledge_base, datasource):
    # delete or add tags in metadata
    metadata = datasource[1]

    # ----- for testing purposes -----
    # output after 1st iteration
    if datasource[0] == 'Parks':
        metadata.update({'park'})
    elif datasource[0] == 'ImportantTrees':
        metadata.update({'species', 'tree'})
    elif datasource[0] == 'ParkSpecimenTrees':
        metadata.update({'park', 'genus', 'species', 'tree'})
    # --------------------------------
    return

def cluster_concepts_in_knowledge_base(knowledge_base):
    # perform schema matching between knowledge base entities
    # perform clustering of entities
    # update synonym and concept hierarchy in knowledge base

    # ----- for testing purposes -----
    # output after 1st iteration
    cluster = {{'tree', 'trees'}, {'park', 'parks'}, {'species', 'genus'}}
    hierarchy = {('species', 'is-child-of', 'genus'), ('genus', 'is-parent-of', 'species')}
    # --------------------------------

    # update the knowledge base

    return

def metadata_enhancer(datasources, knowledge_base, coverage_threshold, covered_by_concepts_init, score_threshold):
    # initialize the knowledge base
    set_of_concepts = collect_concepts_from_tags(datasources)
    merge_concepts_into_knowledge_base(knowledge_base, set_of_concepts, None)
    covered_by_concepts = covered_by_concepts_init # contains each datasource with value False
    coverage_percentage = coverage_threshold

    while not covered_by_concepts:
        for datasource in datasources:
            correspondences_with_score = hybrid_schema_matching(knowledge_base, datasource)
            set_of_concepts_with_score = build_concept(datasource, correspondences_with_score, score_threshold)

            merge_concepts_into_knowledge_base(knowledge_base, set_of_concepts_with_score, datasource)
            update_datasource_with_metadata(knowledge_base, datasource)

            if check_dataset_covered_by_concepts(knowledge_base, datasource) < coverage_percentage:
                generate_new_concepts(knowledge_base, datasource)
            else:
                covered_by_concepts[datasource] = True

        cluster_concepts_in_knowledge_base(knowledge_base)
        # if all covered_by_concepts True for all datasources, then break

    # delete unused concepts in knowledge base
    return

covered_by_concepts_init = {'Parks': False, 'ImportantTrees': False, 'ParkSpecimenTrees': False}
metadata_enhancer(data.DATASOURCES, data.KNOWLEDGE_BASE, 95, covered_by_concepts_init, 0.1)