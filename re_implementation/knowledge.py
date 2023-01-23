import pathlib


class KnowledgeGenerator(object):
    """ class to treat the knowledge generation """

    def __init__(self, model, args):
        super(KnowledgeGenerator, self).__init__()
        self.data = model.data
        self.dataset = args.dataset

        self.reset_files()

    @property
    def knowledge(self):
        self.generate_knowledge()
        return f'{self.dataset}_knowledge_base'

    def reset_files(self):
        """ Deletes knowledge base and datastats file that might
        still be in directory from previous runs """
        know_base = pathlib.Path(f'{self.dataset}_knowledge_base')
        stats = pathlib.Path(f'{self.dataset}_stats')
        if know_base.is_file():
            know_base.unlink()
            print(f'{self.dataset} knowledge base deleted')
        if stats.is_file():
            stats.unlink()
            print(f'{self.dataset} stats deleted')

    def generate_knowledge(self):
        """
        creates the knowledge file based on unary predicates = document classes
        cite is binary predicate
        num_classes int
        """
        assert hasattr(self.data, 'num_classes')

        class_list = []
        for i in list(range(self.data.num_classes)):
            class_list += ['class_' + str(i)]

        if not class_list:
            UserWarning('Empty knowledge base. Choose other filters to keep more clauses ')
            return ''

        # Generate knowledge
        kb = ''

        # List of predicates
        for c in class_list:
            kb += c + ','

        kb = kb[:-1] + '\nLink\n\n'

        # No unary clauses
        kb = kb[:-1] + '\n>\n'

        # Binary clauses
        # eg: nC(x),nCite(x.y),C(y)
        for c in class_list:
            kb += '_:n' + c + '(x),nLink(x.y),' + c + '(y)\n'

        with open(f'{self.dataset}_knowledge_base', 'w') as kb_file:
            kb_file.write(kb)

