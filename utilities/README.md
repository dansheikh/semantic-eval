.##Requirements:
* Python 3
* xml.sax
* xml_utils.py
* auto_ann.py

## Script usage:
* auto_ann.py: automated annotation script. Usage: ```auto_ann.py -s=$'\t' [sample annotations file] [xml files dir] [output file path] > [ner tag file] ```
    * -s, --sep (optional; default: '\s'): annotation sample file separator. 
    * annotation sample file (required): file containing word/annotation pairs, where each pair is on a distinct line. 
    * xml directory (required): directory containing xml files.
    * output file path (required): destination file for annotations.
* Example: auto_ann.py -s=$'\t' ./ann_sample.txt ./train/ ./auto_annotated.txt > ner_tags.json

