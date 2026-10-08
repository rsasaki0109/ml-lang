#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Fixed(usize),
    Named(String),
}
pub type Shape = Vec<Dim>;

#[derive(Clone, Debug)]
pub enum Expr {
    Number(f64),
    Array(Vec<Expr>),
    Var(String),
    Binary(char, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}
#[derive(Clone, Debug)]
pub enum Stmt {
    Bind {
        name: String,
        shape: Option<Shape>,
        value: Expr,
        param: bool,
    },
    Repeat(usize, Vec<Stmt>),
    Call(String, Vec<Expr>),
}
#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub args: Vec<(String, Shape)>,
    pub result: Shape,
    pub locals: Vec<Stmt>,
    pub body: Expr,
}
#[derive(Debug)]
pub struct Ast {
    pub functions: Vec<Function>,
    pub body: Vec<Stmt>,
}
#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Id(String),
    Number(f64),
    Sym(char),
    Arrow,
    Eof,
}
#[derive(Clone, Debug)]
struct Token {
    kind: Kind,
    line: usize,
    column: usize,
}
fn lex(source: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = source.chars().collect();
    let (mut i, mut line, mut column) = (0, 1, 1);
    let mut tokens = vec![];
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            i += 1;
            line += 1;
            column = 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            column += 1;
            continue;
        }
        if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
                column += 1;
            }
            continue;
        }
        let start = i;
        let col = column;
        let kind = if c.is_ascii_alphabetic() || c == '_' {
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            Kind::Id(chars[start..i].iter().collect())
        } else if c.is_ascii_digit()
            || (c == '.' && chars.get(i + 1).is_some_and(|x| x.is_ascii_digit()))
        {
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                i += 1;
                if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let s: String = chars[start..i].iter().collect();
            let n: f64 = s
                .parse()
                .map_err(|_| format!("{line}:{col}: invalid number {s}"))?;
            if !n.is_finite() {
                return Err(format!("{line}:{col}: number must be finite"));
            }
            Kind::Number(n)
        } else if c == '-' && chars.get(i + 1) == Some(&'>') {
            i += 2;
            Kind::Arrow
        } else if "(){}[],:;=+-*/@<>".contains(c) {
            i += 1;
            Kind::Sym(c)
        } else {
            return Err(format!("{line}:{col}: unexpected character {c:?}"));
        };
        column += i - start;
        tokens.push(Token {
            kind,
            line,
            column: col,
        });
    }
    tokens.push(Token {
        kind: Kind::Eof,
        line,
        column,
    });
    Ok(tokens)
}
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
}
impl Parser {
    fn kind(&self) -> &Kind {
        &self.tokens[self.pos].kind
    }
    fn error(&self, message: &str) -> String {
        let t = &self.tokens[self.pos];
        format!("{}:{}: {message}", t.line, t.column)
    }
    fn sym(&mut self, c: char) -> bool {
        if self.kind() == &Kind::Sym(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.sym(c) {
            Ok(())
        } else {
            Err(self.error(&format!("expected '{c}'")))
        }
    }
    fn id(&mut self) -> Result<String, String> {
        if let Kind::Id(s) = self.kind().clone() {
            self.pos += 1;
            Ok(s)
        } else {
            Err(self.error("expected identifier"))
        }
    }
    fn keyword(&mut self, s: &str) -> Result<(), String> {
        if self.kind() == &Kind::Id(s.into()) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error(&format!("expected '{s}'")))
        }
    }
    fn integer(&mut self) -> Result<usize, String> {
        if let Kind::Number(n) = self.kind() {
            if *n >= 0.0 && n.fract() == 0.0 && *n <= 1_000_000.0 {
                let v = *n as usize;
                self.pos += 1;
                return Ok(v);
            }
        }
        Err(self.error("expected integer in 0..1000000"))
    }
    fn shape(&mut self) -> Result<Shape, String> {
        self.keyword("Tensor")?;
        if self.sym('<') {
            let dtype = self.id()?;
            self.expect(',')?;
            let device = self.id()?;
            self.expect('>')?;
            if dtype != "f64" || device != "cpu" {
                return Err(self.error(
                    "v0.1 supports only Tensor<f64,cpu>; other dtypes/devices are not implemented",
                ));
            }
        }
        self.expect('[')?;
        let mut dims = vec![];
        if !self.sym(']') {
            loop {
                let dim = if matches!(self.kind(), Kind::Id(_)) {
                    Dim::Named(self.id()?)
                } else {
                    let n = self.integer()?;
                    if n == 0 {
                        return Err(self.error("zero dimensions are unsupported"));
                    }
                    Dim::Fixed(n)
                };
                dims.push(dim);
                if self.sym(']') {
                    break;
                }
                self.expect(',')?;
            }
        }
        if dims.len() > 8 {
            return Err(self.error("maximum rank is 8"));
        }
        Ok(dims)
    }
    fn args(&mut self) -> Result<Vec<Expr>, String> {
        self.expect('(')?;
        let mut args = vec![];
        if !self.sym(')') {
            loop {
                args.push(self.expr(0)?);
                if self.sym(')') {
                    break;
                }
                self.expect(',')?;
            }
        }
        Ok(args)
    }
    fn expr(&mut self, min: u8) -> Result<Expr, String> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(self.error("expression nesting exceeds 128"));
        }
        let mut lhs = if self.sym('-') {
            Expr::Binary('*', Box::new(Expr::Number(-1.0)), Box::new(self.expr(30)?))
        } else if self.sym('(') {
            let e = self.expr(0)?;
            self.expect(')')?;
            e
        } else if self.sym('[') {
            let mut xs = vec![];
            if !self.sym(']') {
                loop {
                    xs.push(self.expr(0)?);
                    if self.sym(']') {
                        break;
                    }
                    self.expect(',')?;
                }
            }
            Expr::Array(xs)
        } else {
            match self.kind().clone() {
                Kind::Number(n) => {
                    self.pos += 1;
                    Expr::Number(n)
                }
                Kind::Id(s) => {
                    self.pos += 1;
                    if self.kind() == &Kind::Sym('(') {
                        Expr::Call(s, self.args()?)
                    } else {
                        Expr::Var(s)
                    }
                }
                _ => return Err(self.error("expected expression")),
            }
        };
        loop {
            let op = match self.kind() {
                Kind::Sym(c) if "+-*/@".contains(*c) => *c,
                _ => break,
            };
            let prec = if "+-".contains(op) { 10 } else { 20 };
            if prec < min {
                break;
            }
            self.pos += 1;
            let rhs = self.expr(prec + 1)?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        self.depth -= 1;
        Ok(lhs)
    }
    fn block(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect('{')?;
        self.depth += 1;
        if self.depth > 128 {
            return Err(self.error("block nesting exceeds 128"));
        }
        let mut body = vec![];
        while !self.sym('}') {
            if self.kind() == &Kind::Eof {
                return Err(self.error("unterminated block"));
            }
            body.push(self.stmt()?);
        }
        self.depth -= 1;
        Ok(body)
    }
    fn stmt(&mut self) -> Result<Stmt, String> {
        let name = self.id()?;
        if name == "let" || name == "param" {
            let var = self.id()?;
            let shape = if self.sym(':') {
                Some(self.shape()?)
            } else {
                None
            };
            self.expect('=')?;
            let value = self.expr(0)?;
            self.expect(';')?;
            Ok(Stmt::Bind {
                name: var,
                shape,
                value,
                param: name == "param",
            })
        } else if name == "repeat" {
            let n = self.integer()?;
            Ok(Stmt::Repeat(n, self.block()?))
        } else {
            let args = self.args()?;
            self.expect(';')?;
            Ok(Stmt::Call(name, args))
        }
    }
    fn function(&mut self) -> Result<Function, String> {
        self.keyword("fn")?;
        let name = self.id()?;
        self.expect('(')?;
        let mut args = vec![];
        if !self.sym(')') {
            loop {
                let a = self.id()?;
                self.expect(':')?;
                args.push((a, self.shape()?));
                if self.sym(')') {
                    break;
                }
                self.expect(',')?;
            }
        }
        if self.kind() != &Kind::Arrow {
            return Err(self.error("expected '->'"));
        }
        self.pos += 1;
        let result = self.shape()?;
        self.expect('{')?;
        let mut locals = vec![];
        while self.kind() == &Kind::Id("let".into()) {
            locals.push(self.stmt()?);
        }
        if self.kind() != &Kind::Id("return".into()) {
            return Err(
                self.error("function body requires let declarations followed by a final return")
            );
        }
        self.keyword("return")?;
        let body = self.expr(0)?;
        self.expect(';')?;
        self.expect('}')?;
        Ok(Function {
            name,
            args,
            result,
            locals,
            body,
        })
    }
}
pub fn parse(source: &str) -> Result<Ast, String> {
    let mut p = Parser {
        tokens: lex(source)?,
        pos: 0,
        depth: 0,
    };
    let mut ast = Ast {
        functions: vec![],
        body: vec![],
    };
    while p.kind() != &Kind::Eof {
        if p.kind() == &Kind::Id("fn".into()) {
            ast.functions.push(p.function()?);
        } else {
            ast.body.push(p.stmt()?);
        }
    }
    Ok(ast)
}
