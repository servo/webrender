use std::str::CharIndices;
// A crapy parser for parsing strings like "translate(1, 3)"

pub fn parse_function(s: &str) -> (&str, Vec<&str>) {
    // XXX: This it not particular easy to read. Sorry.
    struct Parser<'a> {
        itr: CharIndices<'a>,
        start: usize,
        o: Option<<CharIndices<'a> as Iterator>::Item>,
    }
    impl<'a> Parser<'a> {
        fn skip_whitespace(&mut self) {
            while let Some(k) = self.o {
                if !k.1.is_whitespace() {
                    break;
                }
                self.start = k.0 + 1;
                self.o = self.itr.next();
            }

        }
    }
    let mut c = s.char_indices();
    let o = c.next();
    let mut p = Parser{itr: c, start: 0, o: o};

    p.skip_whitespace();

    let mut end = p.start;
    while let Some(k) = p.o {
        if !k.1.is_alphabetic() {
            break;
        }
        end = k.0 + 1;
        p.o = p.itr.next();
    }

    let name = &s[p.start..end];
    let mut args = Vec::new();

    p.skip_whitespace();

    if let Some(k) = p.o {
        if !(k.1 == '(') {
            return (name, args);
        }
        p.start = k.0 + 1;
        p.o = p.itr.next();
    }

    loop {
        p.skip_whitespace();

        let mut end = p.start;
        while let Some(k) = p.o {
            if !k.1.is_alphanumeric() {
                break;
            }
            end = k.0 + 1;
            p.o = p.itr.next();
        }

        args.push(&s[p.start..end]);

        p.skip_whitespace();

        if let Some(k) = p.o {
            if k.1 == ')' {
                break;
            }
        }
        if let Some(k) = p.o {
            if !(k.1 == ',') {
                break;
            }
            p.start = k.0 + 1;
            p.o = p.itr.next();
        }

    }
    (name, args)
}

#[test]
fn test() {
    assert!(parse_function("rotate(40)").0 == "rotate");
    assert!(parse_function("  rotate(40)").0 == "rotate");
    assert!(parse_function("  rotate  (40)").0 == "rotate");
    assert!(parse_function("  rotate  (  40 )").1[0] == "40");
}
