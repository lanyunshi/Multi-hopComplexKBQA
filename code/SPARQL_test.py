import json
from SPARQLWrapper import SPARQLWrapper, JSON
import re
from collections import defaultdict

const2rel = {'largest': """ns:location.location.area ns:topic_server.population_number""",
             'biggest': """ns:location.location.area ns:topic_server.population_number""",
             'most': """ns:location.location.area ns:topic_server.population_number ns:military.casualties.lower_estimate
               ns:location.religion_percentage.percentage ns:geography.river.discharge ns:aviation.airport.number_of_runways""",
             'major': """ns:location.location.area ns:topic_server.population_number ns:military.casualties.lower_estimate
               ns:location.religion_percentage.percentage ns:geography.river.discharge ns:aviation.airport.number_of_runways""",
             'predominant': """ns:location.religion_percentage.percentage""",
             'warmest': """ns:travel.travel_destination_monthly_climate.average_max_temp_c""",
             'tallest': """ns:architecture.structure.height_meters"""}
special1hoprel = set(["ns:base.aareas.schema.administrative_area.short_name",
                      "ns:base.schemastaging.context_name.official_name",
                      "ns:type.object.name",
                      "ns:base.schemastaging.context_name.nickname"])

SPARQLPATH = "your/path/to/database"

def test():
    try:
        sparql = SPARQLWrapper(SPARQLPATH) #    PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>
        sparql_txt = """PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT distinct ?name3
    WHERE {
    ns:m.0k2kfpc ns:award.award_nominated_work.award_nominations ?e1.
    ?e1 ns:award.award_nomination.award_nominee ns:m.02pbp9.
    ns:m.02pbp9 ns:people.person.spouse_s ?e2.
    ?e2 ns:people.marriage.spouse ?e3.
    ?e2 ns:people.marriage.from ?e4.
    ?e3 ns:type.object.name ?name3
    MINUS{?e2 ns:type.object.name ?name2}
    }
        """
        #print(sparql_txt)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        print(results)
    except:
        print('Your database is not installed properly !!!')

#?c ns:common.topic.notable_types ns:m.0k2kfpc .

# ns:m.0k2kfpc ?r1 ?k.
# ?k ?r2 ?c .
# ?c ns:people.person.spouse_s ?y .
# ?y ns:people.marriage.spouse ?x .
# ?y ns:people.marriage.type_of_union ns:m.04ztj .
# ?y ns:people.marriage.to ?sk0 .
# ?c ?r ?x2 .
def test_sk0(sparql_txt):
    sparql = SPARQLWrapper(SPARQLPATH) #    PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>
    #print(sparql_txt)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    entity = set()
    try:
        results = sparql.query().convert()
        #print(results)
        if results['results']['bindings']:
            for e in results['results']['bindings']:
                entity.add(e['sk0']['value'].split('/ns/')[-1])
        return list(entity)[0]
    except:
        return None

def SQL_name2entity(name):
    entity = set()
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery("""PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?e WHERE {?e ?r "%s"@en}
    """ %(name))
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results['results']['bindings']:
            for e in results['results']['bindings']:
                entity.add(e['e']['value'].split('/ns/')[-1])
    except:
        pass
    return entity

def SQL_name2type(names):
    entity = set()
    query = '\n'.join(["""?e%s ns:type.object.name "%s"@en. \n?e%s ns:common.topic.notable_types ?t.""" %(name_idx, name, name_idx) for name_idx, name in enumerate(names)])
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql_txt = """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?t WHERE {%s}""" %(query)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results['results']['bindings']:
            for e in results['results']['bindings']:
                entity.add(e['t']['value'].split('/ns/')[-1])
    except:
        pass
    return entity

def SQL_entity2name(e):
    if not re.search('^[mg]\.', e): return e
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery("""PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t WHERE {ns:%s ns:type.object.name ?t.}
    """ %(e))
    # ?x8 ns:type.object.type ?x9 ?x8 ns:type.object.name ?x9.
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        name = results['results']['bindings'][0]['t']['value'] if results['results']['bindings'] else '[UNK]'
    except:
        name = '[UNK]'
    return name

def SQL_entity2type(e):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery("""PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t WHERE {ns:%s ns:common.topic.notable_types ?t.}
    """ %(e))
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        name = results['results']['bindings'][0]['t']['value'].split('/ns/')[-1] if results['results']['bindings'] else None
    except:
        name = None
    return name

def SQL_hr2t(h, r):
    ts = set()
    if re.search('^[mg]\.', h):
        slot = 'ns:%s ns:%s ?t' %(h, r)
    elif re.search('^[mg]\.', r):
        slot = '?t ns:%s ns:%s' %(h, r)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery("""PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t WHERE {%s}limit 30
    """ %slot)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results['results']['bindings']:
            for t in results['results']['bindings']:
                ts.add(t['t']['value'].split('/')[-1])
    except:
        ts = set()
    return list(ts)

def SQL_e0r0r1_e1(p):
    e0, r0, r1 = p
    ts = set()
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery("""PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t1, ?t2 WHERE {ns:%s ns:%s ?t1.\n?t1 ns:%s ?t2}limit 100
    """ %(e0, r0, r1))
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results['results']['bindings']:
            for t in results['results']['bindings']:
                t1 = t['t1']['value'].split('/')[-1] if re.search('^http', t['t1']['value']) else t['t1']['value']
                ts.add((e0, r0, t1))
                t2 = t['t2']['value'].split('/')[-1] if re.search('^http', t['t2']['value']) else t['t2']['value']
                ts.add((t1, r1, t2))
    except:
        ts = set()
    return list(ts)

def SQL_e0r0r1e1_e2(p):
    e0, r0, r1, e1 = p
    ts = set()
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery("""PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t1 WHERE {ns:%s ns:%s ?t1.\n?t1 ns:%s ns:%s}limit 100
    """ %(e0, r0, r1, e1))
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results['results']['bindings']:
            for t in results['results']['bindings']:
                t1 = t['t1']['value'].split('/')[-1] if re.search('^http', t['t1']['value']) else t['t1']['value']
                ts.add((e0, r0, t1))
                ts.add((t1, r1, e1))
    except:
        ts = set()
    return list(ts)

def SQL_e0r0r1r2e2_e3(p):
    e0, r0, r1, r2, e2= p
    ts = set()
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery("""PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t1, ?t2 WHERE {ns:%s ns:%s ?t1.\n?t1 ns:%s ?t2.\n?t2 ns:%s ns:%s}limit 100
    """ %(e0, r0, r1, r2, e2))
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results['results']['bindings']:
            for t in results['results']['bindings']:
                t1 = t['t1']['value'].split('/')[-1] if re.search('^http', t['t1']['value']) else t['t1']['value']
                ts.add((e0, r0, t1))
                t2 = t['t2']['value'].split('/')[-1] if re.search('^http', t['t2']['value']) else t['t2']['value']
                ts.add((t1, r1, t2))
                ts.add((t2, r2, e2))
    except:
        ts = set()
    return list(ts)

def check_dummyentity(p):
    e0, r0, r1 = p
    ts = False
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery("""PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT ?t1, ?t2 WHERE {ns:%s ns:%s ?t1.\n?t1 ns:%s ?t2.\nMINUS {?t1 ns:type.object.name ?name.}.}LIMIT 10
    """ %(e0, r0, r1))
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results['results']['bindings']:
            ts = True
    except:
        pass
    return ts

def form_trips(p_tokens):
    t_idx, d_idx = 0, 0
    trips = ()
    for p_token in p_tokens:
        trip = ()
        if len(p_token) > 3: p_token = p_token[1:]
        for p_idx, e in enumerate(p_token):
            e = 'ns:%s' %e if (re.search('^[mg]\.', e) or len(e.split('.')) > 2) else "'%s'^^xsd:date" %e if (e.isdigit() and int(e) < 2100 or re.search('\d-\d', e)) else e
            if re.search('^\?e', e): t_idx = int(re.findall('\d+', e)[0])
            if re.search('^\?d', e): d_idx = int(re.findall('\d+', e)[0])
            trip += (e, )
        trips += (' '.join(trip), )
    if d_idx < t_idx - 1: d_idx = 0 # if d is too far, unvalid it
    return trips, t_idx, d_idx

def SQL_query(p):
    new_p, p = [], p.split()
    s = 0
    while s < len(p):
        if p[s] not in ['last', 'first']:
            l = 3
        else:
            l = 4
        new_p += [p[s: s+l]]
        s = s+l
    p = tuple(new_p)
    kbs, answer = defaultdict(set), set()
    trips, t_idx, _ = form_trips(p)
    topic = re.findall('^ns\:[mg]\.[^ ]+', trips[0])[0]
    trips = '.\n'.join(query) if t_idx == 0 else '.\n'.join(trips)
    retu = '?e%s' %(t_idx)
    const = "FILTER (?e%s!=%s)\nFILTER (!isLiteral(?e%s) OR lang(?e%s) = '' OR langMatches(lang(?e%s), 'en'))" %((t_idx, topic)+(t_idx, )*3)
    sparql_txt = """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT %s WHERE {%s\n%s}""" %(retu, const, trips) # Q: the order of the query matters ?!
    print(sparql_txt)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        #print(len(results['results']['bindings']))
        if results['results']['bindings']:
            for t in results['results']['bindings']:
                t = t['e%s' %(t_idx)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx)]['value']) else t['e%s' %(t_idx)]['value']
                answer.add(t)
    except:
        pass
    return answer

def SQL_1hop(p, QUERY=None):
    kbs, sparql_txts = defaultdict(set), set()
    trips, t_idx, _ = form_trips(p)
    topic = re.findall('^ns\:[mg]\.[^ ]+', trips[0])[0]
    query = (' '.join(['%s' %topic, '?r', '?e%s' %(t_idx+1)]), ) if t_idx == 0 else (' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )
    trips = '.\n'.join(query) if t_idx == 0 else '.\n'.join(trips + query)
    retu = ', '.join(['?r', '?e%s' %(t_idx+1)])
    const = "FILTER (?e%s!=%s)\nFILTER (!isLiteral(?e%s) OR lang(?e%s) = '' OR langMatches(lang(?e%s), 'en'))" %((t_idx+1, topic)+(t_idx+1, )*3)
    for const1_idx, const1 in enumerate(["?e%s ns:type.object.name ?name." %(t_idx+1), "FILTER (datatype(?e%s) in (xsd:date, xsd:gYear))" %(t_idx+1), "VALUES ?r {%s}" %(' '.join(special1hoprel))]): #
        '''If const1_idx > 1, consider the name-mentioned relation'''
        if const1_idx > 1: const = ""
        sparql_txt = """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT %s WHERE {%s\n%s\n%s}""" %(retu, const, const1, trips) # Q: the order of the query matters ?!
        #print(sparql_txt)
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        try:
            if (QUERY is not None) and sparql_txt in QUERY: return kbs, sparql_txts
            results = sparql.query().convert()
            #print(len(results['results']['bindings']))
            if results['results']['bindings']:
                for t in results['results']['bindings']:
                    r = t['r']['value'].split('/ns/')[-1] if re.search('^http', t['r']['value']) else t['r']['value']
                    t = t['e%s' %(t_idx+1)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx+1)]['value']) else t['e%s' %(t_idx+1)]['value']
                    trip = ((p[0][0], r, '?e%s' %(t_idx+1)), ) if t_idx == 0 else (('?e%s' %t_idx, r, '?e%s' %(t_idx+1)), )
                    kbs[trip].add(t)
            sparql_txts.add(sparql_txt)
        except:
            pass
    return kbs, sparql_txts

def SQL_2hop(p, QUERY=None):
    kbs, sparql_txts = defaultdict(set), set()
    trips, t_idx, _ = form_trips(p)
    topic = re.findall('^ns\:[mg]\.[^ ]+', trips[0])[0]
    query = (' '.join(['%s' %topic, '?r', '?d%s' %(t_idx+1)]), ) if t_idx == 0 else (' '.join(['?e%s' %t_idx, '?r', '?d%s' %(t_idx+1)]), )
    query += (' '.join(['?d%s' %(t_idx+1), '?r1', '?e%s' %(t_idx+2)]), )
    trips = '.\n'.join(query) if t_idx == 0 else '.\n'.join(trips + query)
    retu = ', '.join(['?r', '?r1', '?e%s' %(t_idx+2)])
    const = "FILTER (?e%s!=%s)\nFILTER (!isLiteral(?e%s) OR lang(?e%s) = '' OR langMatches(lang(?e%s), 'en'))." %((t_idx+2, topic)+(t_idx+2, )*3)
    const1 = "MINUS {?d%s ns:type.object.name ?name.}." %(t_idx+1)
    sparql_txt = """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT %s WHERE {%s\n%s\n%s}""" %(retu, const, const1, trips)
    #print(sparql_txt)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    try:
        if (QUERY is not None) and sparql_txt in QUERY: return kbs, sparql_txts
        results = sparql.query().convert()
        #print(len(results['results']['bindings']))
        if results['results']['bindings']:
            for t in results['results']['bindings']:
                r = t['r']['value'].split('/ns/')[-1] if re.search('^http', t['r']['value']) else t['r']['value']
                r1 = t['r1']['value'].split('/ns/')[-1] if re.search('^http', t['r1']['value']) else t['r1']['value']
                t = t['e%s' %(t_idx+2)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx+2)]['value']) else t['e%s' %(t_idx+2)]['value']
                trip = ((p[0][0], r, '?d%s' %(t_idx+1)), ('?d%s' %(t_idx+1), r1, '?e%s' %(t_idx+2))) if t_idx == 0 else (('?e%s' %t_idx, r, '?d%s' %(t_idx+1)), ('?d%s' %(t_idx+1), r1, '?e%s' %(t_idx+2)))
                kbs[trip].add(t)
        sparql_txts.add(sparql_txt)
    except:
        pass
    return kbs, sparql_txts

def SQL_1hop_reverse(p, const_entities, QUERY=None):
    kbs, const1, sparql_txts = defaultdict(set), '', set()
    raw_trips, t_idx, d_idx = form_trips(p)
    if re.search('^[mg]\.', list(const_entities)[0]):
        const_type = 'mid'
        const_entiti = ['ns:%s' %e for e in const_entities]
        const = "VALUES ?e%s {%s}" %(t_idx+1, ' '.join(sorted(const_entiti)))
        queries = [(' '.join(['?e%s' %(t_idx+1), '?r', '?e%s' %t_idx]), ), (' '.join(['?e%s' %(t_idx+1), '?r', '?d%s' %d_idx]), )] if d_idx else [(' '.join(['?e%s' %(t_idx+1), '?r', '?e%s' %t_idx]), )]
        #queries = [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), ), (' '.join(['?d%s' %d_idx, '?r', '?e%s' %(t_idx+1)]), )] if d_idx else [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )]
    elif re.search('^\d', list(const_entities)[0]):
        const_type = 'year'
        year = int(list(const_entities)[0]) #  FILTER(datatype(?e%s) in (xsd:date, xsd:gYear) &&
        #const = "?e%s < xsd:date('%s-01-01') && ?e%s >= xsd:date('%s-01-01'))" %(t_idx+1, year+1, t_idx+1, year)
        const = "FILTER(?e%s >= xsd:date('%s-01-01') && ?e%s < xsd:date('%s-01-01'))" %(t_idx+1, year, t_idx+1, year+1)
        queries = [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), ), (' '.join(['?d%s' %d_idx, '?r', '?e%s' %(t_idx+1)]), )] if d_idx else [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )]
    elif list(const_entities)[0] in ['first', 'last', 'current', 'newly']:
        const_type = list(const_entities)[0]
        const = "FILTER (datatype(?e%s) in (xsd:date, xsd:gYear))" %(t_idx+1)
        const1 = "ORDER BY xsd:datetime(?e%s)\nLIMIT 1" %(t_idx+1) if const_type in ['first'] else "ORDER BY DESC(xsd:datetime(?e%s))\nLIMIT 1" %(t_idx+1)
        queries = [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), ), (' '.join(['?d%s' %d_idx, '?r', '?e%s' %(t_idx+1)]), )] if d_idx else [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )]
    elif list(const_entities)[0] in ['largest', 'most', 'predominant', 'biggest', 'major', 'warmest', 'tallest']:
        const_type = list(const_entities)[0]
        const = "VALUES ?r {%s}" %const2rel[const_type]
        const1 = "ORDER BY DESC(xsd:float(?e%s))\nLIMIT 1" %(t_idx+1)
        queries = [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), ), (' '.join(['?d%s' %d_idx, '?r', '?e%s' %(t_idx+1)]), )] if d_idx else [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )]
    elif list(const_entities)[0] in ['daughter', 'son']:
        const_type = list(const_entities)[0]
        const = "VALUES ?e%s {ns:m.05zppz}" %(t_idx+1) if const_type in ['son'] else "VALUES ?e%s {ns:m.02zsn}" %(t_idx+1)
        queries = [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), ), (' '.join(['?d%s' %d_idx, '?r', '?e%s' %(t_idx+1)]), )] if d_idx else [(' '.join(['?e%s' %t_idx, '?r', '?e%s' %(t_idx+1)]), )]
    else:
        raise Exception('SQL_1hop_reverse has wrong constraint format %s' %str(const_entities))
    for q_idx, query in enumerate(queries):
        trips = '.\n'.join(raw_trips + query)
        retu = ', '.join(['?e%s' %(t_idx+1), '?r', '?e%s' %t_idx])
        sparql_txt = """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s}%s""" %(retu, trips, const, const1)
        #print(sparql_txt)
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        try:
            if (QUERY is not None) and sparql_txt in QUERY: continue
            results = sparql.query().convert()
            if results['results']['bindings']:
                for t in results['results']['bindings']:
                    h = t['e%s' %(t_idx+1)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx+1)]['value']) else t['e%s' %(t_idx+1)]['value']
                    r = t['r']['value'].split('/ns/')[-1] if re.search('^http', t['r']['value']) else t['r']['value']
                    t = t['e%s' %(t_idx)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx)]['value']) else t['e%s' %(t_idx)]['value']
                    if const_type in ['mid']:
                        #kbs[((h, r, '?e%s' %t_idx), )].add(t) if q_idx==0 else kbs[((h, r, '?d%s' %d_idx), )].add(t)
                        kbs[(('?e%s' %t_idx, r, h), )].add(t) if q_idx==0 else kbs[(('?d%s' %d_idx, r, h), )].add(t)
                    elif const_type in ['son', 'daughter']:
                        kbs[((const_type, '?e%s' %t_idx, r, h), )].add(t) if q_idx==0 else kbs[((const_type, '?d%s' %d_idx, r, h), )].add(t)
                    elif const_type in ['year']:
                        kbs[(('?e%s' %t_idx, r, h), )].add(t) if q_idx==0 else kbs[(('?d%s' %d_idx, r, h), )].add(t)
                    elif const_type in ['first', 'last', 'current']:
                        kbs[((const_type, '?e%s' %t_idx, r, h), )].add(t) if q_idx==0 else kbs[((const_type, '?d%s' %d_idx, r, h), )].add(t)
                    elif const_type in ['largest', 'most', 'predominant', 'biggest', 'major', 'warmest', 'tallest']:
                        kbs[((const_type, '?e%s' %t_idx, r, h), )].add(t) if q_idx==0 else kbs[((const_type, '?d%s' %d_idx, r, h), )].add(t)
            sparql_txts.add(sparql_txt)
        except:
            pass
    return kbs, sparql_txts

def SQL_2hop_reverse(p, const_entities, QUERY=None):
    kbs, const1, sparql_txts = defaultdict(set), '', set()
    raw_trips, t_idx, d_idx = form_trips(p)
    if re.search('^[mg]\.', list(const_entities)[0]):
        const_type = 'mid'
        const_entiti = ['ns:%s' %e for e in const_entities]
        const = "VALUES ?e%s {%s}" %(t_idx+2, ' '.join(sorted(const_entiti)))
        if d_idx:
            queries = [('?d%s ?r ?e%s' %(t_idx+1, t_idx), '?e%s ?r2 ?d%s' %(t_idx+2, t_idx+1), 'MINUS {?d%s ns:type.object.name ?name.}.' %(t_idx+1))]
            queries += [('?d%s ?r ?d%s' %(t_idx+1, d_idx), '?e%s ?r2 ?d%s' %(t_idx+2, t_idx+1), 'MINUS {?d%s ns:type.object.name ?name.}.' %(t_idx+1))]
        else:
            queries = [('?d%s ?r ?e%s' %(t_idx+1, t_idx), '?e%s ?r2 ?d%s' %(t_idx+2, t_idx+1), 'MINUS {?d%s ns:type.object.name ?name.}.' %(t_idx+1))]
    else:
        raise Exception('SQL_2hop_reverse has wrong constraint format %s' %str(const_entities))
    for q_idx, query in enumerate(queries):
        trips = '.\n'.join(raw_trips + query)
        retu = '?e%s, ?r2, ?r, ?e%s' %(t_idx+2, t_idx)
        sparql_txt = """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT DISTINCT %s WHERE {%s\n%s}%s""" %(retu, trips, const, const1)
        #print(sparql_txt)
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        try:
            if (QUERY is not None) and sparql_txt in QUERY: continue
            results = sparql.query().convert()
            if results['results']['bindings']:
                for t in results['results']['bindings']:
                    h = t['e%s' %(t_idx+2)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx+2)]['value']) else t['e%s' %(t_idx+2)]['value']
                    r2 = t['r2']['value'].split('/ns/')[-1] if re.search('^http', t['r2']['value']) else t['r2']['value']
                    r = t['r']['value'].split('/ns/')[-1] if re.search('^http', t['r']['value']) else t['r']['value']
                    t = t['e%s' %(t_idx)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx)]['value']) else t['e%s' %(t_idx)]['value']
                    if const_type in ['mid']:
                        kbs[((h, r2, 'd%s' %(t_idx+1)), ('d%s' %(t_idx+1), r, '?e%s' %t_idx))].add(t) if q_idx == 0 else kbs[((h, r2, 'd%s' %(t_idx+1)), ('d%s' %(t_idx+1), r, '?d%s' %d_idx))].add(t)
            sparql_txts.add(sparql_txt)
        except:
            pass
    return kbs, sparql_txts

def SQL_1hop_type(p, const_entities, QUERY=None):
    kbs, sparql_txts = defaultdict(set), set()
    raw_trips, t_idx, _ = form_trips(p)
    const_entiti = ['ns:%s' %e for e in const_entities]
    query = (' '.join(['?e%s' %t_idx, 'ns:common.topic.notable_types', '?e%s' %(t_idx+1)]), )
    trips = '.\n'.join(raw_trips + query)
    retu = ', '.join(['?e%s' %t_idx, '?e%s' %(t_idx+1)])
    const = "VALUES ?e%s {%s}" %(t_idx+1, ' '.join(sorted(const_entiti)))
    sparql_txt = """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT %s WHERE {%s\n%s}""" %(retu, const, trips)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    try:
        if (QUERY is not None) and sparql_txt in QUERY: return kbs, sparql_txts
        results = sparql.query().convert()
        for t in results['results']['bindings']:
            h = t['e%s' %(t_idx+1)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx+1)]['value']) else t['e%s' %(t_idx+1)]['value']
            t = t['e%s' %(t_idx)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx)]['value']) else t['e%s' %(t_idx)]['value']
            kbs[(('?e%s' %t_idx, 'common.topic.notable_types', h), )].add(t)
        sparql_txts.add(sparql_txt)
    except:
        pass
    return kbs, sparql_txts

def retrieve_answer(p):
    answers = set()
    trips, t_idx, _ = form_trips(p)
    retu = '?e%s' %(t_idx)
    trips = '.\n'.join(trips)
    sparql_txt = """PREFIX ns:<http://rdf.freebase.com/ns/>\nSELECT %s WHERE {%s}""" %(retu, trips)
    # print(sparql_txt)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    if results['results']['bindings']:
        for t in results['results']['bindings']:
            t = t['e%s' %(t_idx)]['value'].split('/ns/')[-1] if re.search('^http', t['e%s' %(t_idx)]['value']) else t['e%s' %(t_idx)]['value']
            answers.add(t)

    return answers

if __name__ == '__main__': #m.0bpn2j', ['people.person.spouse_s', 'people.marriage.spouse
    #print(SQL_query("m.0f8l9c location.location.adjoin_s ?d1 ?d1 location.adjoining_relationship.adjoins ?e2"))
    #print([k for k in SQL_1hop((('m.0j0k', ), ), None)[0].items() if 'countries_within' in k[0][0][1]]) # "m.0dl567": 0.8043850674528722,
    test()# , ('?d1', 'sports.sports_team_roster.from', '2003')
    #print(SQL_1hop((('m.0wz6jjk', 'common.topic.notable_for', '?d1'), ('?d1', 'common.notable_for.notable_object', '?e2'))))
    #print(SQL_name2entity('The Jeff Probst Show'))
