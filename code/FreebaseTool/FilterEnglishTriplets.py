'''

@author: Yunshi Lan

Pre-process Freebase to by removing non-English or non-digital triplets.

'''

# !/usr/bin/env python

import re
import sys


prefixes = re.compile("@")
quotes = re.compile("[\"]")
ns = "http://rdf.freebase.com/ns/"
xml = "http://www.w3.org/2001/XMLSchema"
re_ns_ns = "^\<{0}[mg]\.[^>]+\>\t\<{0}[^>]+\>\t\<{0}[^>]+\>\t.$".format(ns)
re_ns_en = "^\<{0}[mg]\.[^>]+\>\t\<{0}[^>]+\>\t[\'\"](?!/).+[\'\"](?:\@en)?\t\.$".format(ns)
re_ns_xml = "^\<{0}[mg]\.[^>]+\>\t\<{0}[^>]+\>\t.+\<{1}\#[\w]+\>\t.$".format(ns, xml)

line_number = 0
for line in sys.stdin:
    line_number += 1
    # line = line.rstrip().replace(ns, 'ns:').replace(key, 'key:')
    line = line.rstrip()
    if line == "":
        sys.stdout.write('\n')
    elif prefixes.match(line):
        sys.stdout.write(line + '\n')
    elif line[-1] != ".":
        sys.stderr.write("No full stop: skipping line %d\n" % (line_number))
        continue
    # elif len(quotes.findall(line)) % 2 != 0:
    #    sys.stderr.write("Incomplete quotes: skipping line %d\n" %(line_number))
    #    continue
    else:
        parts = line.split("\t")
        if len(parts) != 4 or parts[0].strip() == "" or parts[1].strip() == "" or parts[2].strip() == "":
            sys.stderr.write("n tuple size != 3: skipping line %d\n" % (line_number))
            continue

        if re.search(re_ns_en, line):
            sys.stdout.write(line + "\n")
        elif re.search(re_ns_ns, line):
            sys.stdout.write(line + "\n")
        elif re.search(re_ns_xml, line):
            sys.stdout.write(line + "\n")

    if line_number % 1000000 == 0:
        #sys.stderr.write("{}: {}\n".format(part, line_number))
        sys.stderr.flush()
