#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import xml.sax
import xml_utils


class Passages:
    def __init__(self, xmlpath):
        self._xmlpath = xmlpath
        self._handlers = list()
        self._regex = r"[\w]+|\[|\]|\,|\."
        self._parse_xmls(self._xmlpath)

    def _parse_xml(self, inpath):
        # Create an XMLReader
        parser = xml.sax.make_parser()
        # Turn off namespaces.
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        handler = xml_utils.PubHandler()
        parser.setContentHandler(handler)

        # Parse document.
        parser.parse(inpath)

        return handler

    def _parse_xmls(self, inpath):
        if os.path.isdir(self._xmlpath):
            dir = os.listdir(inpath)

            for file in dir:
                if not file.endswith(".xml"):
                    continue

                self._handlers.append(self._parse_xml(os.path.join(inpath, file)))

        else:
            self._handlers.append(self._parse_xml(self._xmlpath))

    def __iter__(self):
        for handler in self._handlers:
            for idx, text in handler.text.items():
                yield re.findall(self._regex, text)
