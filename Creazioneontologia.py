from owlready2 import *

onto = get_ontology("file://mioVino.rdf").load()


with onto:
    class Wine(Thing):
        pass
    class Winery(Thing):
        pass
    class Red_wine(Wine):
        pass
    class White_wine(Wine):
        pass
    class Sparkling_wine(Wine):
        pass
    class Rose_wine(Wine):
        pass
    class Grape(Thing):
        pass
    class Wine_Region(Thing):
        pass

with onto:
    class is_from_country(Wine_Region >> str):
        pass
    class is_from_region(Wine_Region >> str):
        pass

onto.save()



