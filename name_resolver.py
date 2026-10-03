from __future__ import annotations

from ast_nodes import (Program, Assignment, BinaryExpr, Block, BoolLiteral, CallExpr, CallStmt, Expr, FunctionDecl, 
                       IdentifierExpr, IfStmt, IntLiteral, PrintStmt, ReturnStmt, StringLiteral, UnaryExpr, VarDecl, WhileStmt)

from semantic_errors import SemanticErrorKind, SemanticDiagnostic, SemanticError

from symbols import SymbolKind, Symbol, FunctionSymbol, Scope


def resolve_names(program: Program) -> None:
    """Construa escopos, símbolos e vínculos entre usos e declarações."""

    diagnostics: list[SemanticDiagnostic] = []
    global_functions: dict[str, FunctionSymbol] = {}

    for function in program.functions: # Todas as assinaturas são conhecidas antes de visitar qualquer corpo.
        if function.name in global_functions:
            diagnostics.append(_diag(SemanticErrorKind.DUPLICATE_FUNCTION, function, f"funcao {function.name!r} ja foi declarada")) # O primeiro símbolo permanece como contrato de resolução.
            continue

        symbol = FunctionSymbol(function.name, SymbolKind.FUNCTION,function.return_type, function, tuple(parameter.type for parameter in function.parameters),)
        global_functions[function.name] = symbol
        function.metadata["symbol"] = symbol

    main = global_functions.get("main") # O nome main pertence ao espaço global de funções.
    if main is None:
        diagnostics.append(_diag(SemanticErrorKind.INVALID_MAIN, program, "programa deve declarar int main()",))
    elif (main.type.value != "int" or len(main.parameter_types) != 0):
        diagnostics.append(_diag(SemanticErrorKind.INVALID_MAIN, main.declaration, "main deve ter assinatura int main()",))

    for function in program.functions: # Os corpos são percorridos mesmo quando já existem erros independentes.
        resolver = _FunctionResolver(function, global_functions, diagnostics)
        resolver.visit_block(function.body, None, is_function_body=True)

    if diagnostics:
        raise SemanticError(diagnostics)

class _FunctionResolver:
    def __init__(self, function, functions, diagnostics):
        self.function = function
        self.functions = functions
        self.diagnostics = diagnostics

    def visit_block(self, block: Block, parent: Scope | None, *, is_function_body=False) -> Scope:
        scope = Scope(parent)
        block.metadata["scope"] = scope

        if is_function_body: # Os parâmetros pertencem ao escopo do bloco externo.
            for parameter in self.function.parameters:
                symbol = Symbol(parameter.name, SymbolKind.PARAMETER, parameter.type, parameter, )
                if parameter.name in scope.symbols:
                    self.diagnostics.append(_diag(SemanticErrorKind.DUPLICATE_DECLARATION, parameter, f"nome {parameter.name!r} ja foi declarado neste escopo", ))
                else:
                    scope.symbols[parameter.name] = symbol
                    parameter.metadata["symbol"] = symbol

        for statement in block.statements:
            self.visit_stmt(statement, scope)
        return scope

    def visit_stmt(self, stmt, scope: Scope) -> None:
        if isinstance(stmt, VarDecl): # A declaração entra no escopo antes do inicializador.
            symbol = Symbol(stmt.name, SymbolKind.VARIABLE, stmt.type, stmt)
            if stmt.name in scope.symbols:
                self.diagnostics.append(_diag(SemanticErrorKind.DUPLICATE_DECLARATION, stmt, f"nome {stmt.name!r} ja foi declarado neste escopo", ))
            else:
                scope.symbols[stmt.name] = symbol
                stmt.metadata["symbol"] = symbol
            if stmt.initializer is not None:
                self.visit_expr(stmt.initializer, scope)
            return

        if isinstance(stmt, Assignment):
            self.visit_expr(stmt.target, scope)
            self.visit_expr(stmt.value, scope)
            return

        if isinstance(stmt, CallStmt):
            self.visit_call(stmt.call, scope)
            return

        if isinstance(stmt, IfStmt):
            self.visit_expr(stmt.condition, scope)
            self.visit_block(stmt.then_block, scope)
            if stmt.else_block is not None:
                self.visit_block(stmt.else_block, scope)
            return

        if isinstance(stmt, WhileStmt):
            self.visit_expr(stmt.condition, scope)
            self.visit_block(stmt.body, scope)
            return

        if isinstance(stmt, ReturnStmt):
            if stmt.value is not None:
                self.visit_expr(stmt.value, scope)
            return

        if isinstance(stmt, PrintStmt):
            for item in stmt.items:
                if isinstance(item, Expr):
                    self.visit_expr(item, scope)
            return

        if isinstance(stmt, Block):
            self.visit_block(stmt, scope)
            return

    def visit_expr(self, expr: Expr, scope: Scope) -> None:
        if isinstance(expr, IdentifierExpr):
            symbol = self.lookup(scope, expr.name)
            if symbol is None:
                self.diagnostics.append(_diag(SemanticErrorKind.UNDECLARED_VARIABLE, expr, f"variavel {expr.name!r} nao foi declarada", ))
            else:
                expr.metadata["symbol"] = symbol
            return

        if isinstance(expr, CallExpr):
            self.visit_call(expr, scope)
            return

        if isinstance(expr, UnaryExpr):
            self.visit_expr(expr.operand, scope)
            return

        if isinstance(expr, BinaryExpr):
            self.visit_expr(expr.left, scope)
            self.visit_expr(expr.right, scope)
            return

        if isinstance(expr, (IntLiteral, BoolLiteral)):
            return

    def visit_call(self, call: CallExpr, scope: Scope) -> None:
        function = self.functions.get(call.name)
        if function is None:
            self.diagnostics.append(_diag(SemanticErrorKind.UNDECLARED_FUNCTION, call, f"funcao {call.name!r} nao foi declarada", ))
        else:
            call.metadata["symbol"] = function

        for argument in call.arguments:
            self.visit_expr(argument, scope)

    @staticmethod
    def lookup(scope: Scope, name: str) -> Symbol | None:
        current = scope
        while current is not None:
            symbol = current.symbols.get(name)
            if symbol is not None:
                return symbol
            current = current.parent
        return None

def _diag(kind, node, message):
    return SemanticDiagnostic(kind, message, node.span)
