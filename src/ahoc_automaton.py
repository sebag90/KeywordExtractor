from collections import deque


class State:

    def __init__(self, symbol=None):
        self.children = dict()
        self.root = False
        if symbol is None:
            symbol = "ROOT"
            self.root = True
        self.symbol = symbol
        self.output = list()
        self.fail = None
        self.results = dict()
        self.__counter = 0
        self.counts = dict()

    def __repr__(self):
        return self.symbol

    def __str__(self, level=0):
        ret = "  "*level+repr(self.symbol)+"\n"
        for child in self.children.values():
            ret += child.__str__(level+1)
        return ret

    def reset(self):
        """
        This function delets results, counts and the counter.
        Only the patterns are kept - used to search the same
        patterns in another text. Particulary useful if working
        with many patterns.
        """
        self.results = dict()
        self.__counter = 0
        self.counts = dict()

    def traverse(self, states=None):
        """
        a recursive function to traverse the
        automaton and return a list of states

        Parameters:
            - None (self)

        Returns:
            - list of states in the automaton
        """
        if states is None:
            states = list()

        states.append(self.symbol)

        for child in self.children.values():
            child.traverse(states)

        return states

    def find_next_state(self, char):
        """
        find the next state in the automaton given a char

        Argument:
            - char (charachter): the next charachter to be matched

        Returns:
            - state (object) : if the charachter is a child
                of the current state
            -None: otherwise
        """
        if char in self.children:
            return self.children[char]
        return None

    def add_pattern(self, pattern):
        """
        add a new search pattern to the automaton
        (algorithm 2 from paper)

        Argument:
            - pattern (string): a matching pattern to be added to
                the automaton

        Returns:
            - void: If necessary, new states are created to match
                the new pattern
        """
        this_state = self

        for char in pattern:
            # if a new state for the char already exists, move there
            if this_state.find_next_state(char) is not None:
                this_state = this_state.find_next_state(char)

            # create a new state and add it to current state as a child
            else:
                new_state = State(char)
                this_state.children[char] = new_state
                this_state = new_state

        # accepting state! add pattern to state's output
        this_state.output.append(pattern)

    def fail_connections(self):
        """
        calculate for each state the failure connection
        (algorithm 3 from paper)

        Arguments:
            - None (self)

        Returns:
            - void: creates fail connections between the states
                of the automaton to enable pattern matching
        """
        root = self
        self.fail = root
        queue = deque()

        for node in self.children.values():
            queue.append(node)
            node.fail = root

        while queue:
            r = queue.popleft()

            for child in r.children.values():
                queue.append(child)
                state = r.fail

                # if no new state (and no root because root.fail = root)
                # --> follow fail links
                while (state.find_next_state(child.symbol) is None
                       and state.root is False):
                    state = state.fail

                # set next state as fail state
                child.fail = state.find_next_state(child.symbol)

                # if next state does not existe, set fail to root
                if child.fail is None:
                    child.fail = root

                # carry over the output
                child.output = child.output + child.fail.output

    def find_match(self, line, case_insensitive=False):
        """
        given a string, run it through the automaton to find a match
        (algorithm 1 from the paper)

        Parameters:
            -line (string): the text to be searched for matches
            -case_insentitive (bool): standard = False, if true,
                the matching algorithm ignores case differences
                in line and search patterns

        Returns:
            -void (saves the index where the matches begins in
                   self__results[pattern], index is of type integer)
        """
        if case_insensitive:
            line = line.lower()

        current_state = self
        root = self

        for i, char in enumerate(line):
            # if no new state --> follow fail links
            while (current_state.find_next_state(char) is None
                   and current_state.root is False):
                current_state = current_state.fail

            # go to the next state
            current_state = current_state.find_next_state(char)

            # if next state does not exists, go back to root
            if current_state is None:
                current_state = root
            else:
                # if we are in a terminal state
                # (aka state has output) save result
                for pattern in current_state.output:
                    if pattern not in self.results:
                        self.results[pattern] = list()

                    if pattern not in self.counts:
                        self.counts[pattern] = 0
                    # add counter to i (for multiline input)
                    it = i + self.__counter
                    self.results[pattern].append(it - len(pattern) + 1)
                    self.counts[pattern] += 1

        self.__counter += len(line)

    @classmethod
    def create_automaton(cls, string_list):
        """
        A class method to create and return a complete Aho Corasick Automaton
        given a list of patterns to be added as states

        Parameters:
            - cls (class State):
            - string_list (list of strings): each string is a
                pattern to be matched

        Returns:
            - automaton (object): a complete Aho-Corasick Automaton
              to match the patterns given as argument
        """
        automaton = cls()
        for string in string_list:
            automaton.add_pattern(string)
        automaton.fail_connections()
        return automaton


if __name__ == "__main__":
    text = (
        "The PRADA Christmas Race is a one day knock out series, "
        "based on the seeding from the PRADA ACWS Auckland, NZ and "
        "the last chance for teams to take on the Defender before "
        "the 36th America’s Cup Presented by PRADA."
    )

    patterns = ["PRADA", "on", "PRACHT", "oboe"]

    s = State.create_automaton(patterns)
    s.find_match(text)

    print(f"Text:\n{text}")
    print(f"\nWe want to match these words:\t{patterns}")
    print("First we build the automaton:")
    print(s)
    print("and now we use it to find the indeces:")
    for key in s.results:
        print(f"{key:<7}{s.results[key]}")

    print("\nor count the absolute frequence of the patterns")
    for key in s.counts:
        print(f"{key:<7}{s.counts[key]}")
