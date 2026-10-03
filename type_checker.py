from __future__ import annotations

from ast_nodes import (
    Assignment,
    BinaryExpr,
    BinaryOperator,
    Block,
    BoolLiteral,
    CallExpr,
    CallStmt,
    Expr,
    IdentifierExpr,
    IfStmt,
    IntLiteral,
    PrintStmt,
    Program,
    ReturnStmt,
    Stmt,
    StringLiteral,
    TypeName,
    UnaryExpr,
    UnaryOperator,
    VarDecl,
    WhileStmt,
)
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind
from symbols import FunctionSymbol, Symbol


def check_types(program: Program) -> None:
    """Determine tipos de expressões e valide seus contextos."""

    # Lista para guardar todos os erros encontrados. 
    diagnostics: list[SemanticDiagnostic] = []

    def set_type(expression: Expr, type_: TypeName) -> TypeName: 
        expression.metadata["type"] = type_ 
        return type_

    def expression_type(expression: Expr, allow_void: bool = False) -> TypeName | None: 
        # Literal inteiro -> int 
        if isinstance(expression, IntLiteral): 
            # O literal inteiro precisa estar entre 0 e 2^63 - 1.
            if expression.value < 0 or expression.value > 2**63 - 1:
                diagnostics.append(SemanticDiagnostic(
                    SemanticErrorKind.INTEGER_LITERAL_OUT_OF_RANGE,
                    "literal inteiro fora do intervalo permitido",
                    expression.span,
                    ))
                return None
                        
            return set_type(expression, TypeName.INT) 
        
        # Literal booleano -> bool 
        if isinstance(expression, BoolLiteral):
            return set_type(expression, TypeName.BOOL) 
         
        # Identificador: o name_resolver já colocou o símbolo no metadata. 
        if isinstance(expression, IdentifierExpr): 
            symbol = expression.metadata.get("symbol") 
            if symbol is not None: 
                return set_type(expression, symbol.type) 
            return None 
        
        # Expressão unária
        if isinstance(expression, UnaryExpr):
            tipo_operando = expression_type(expression.operand)
            # Se o operando já tem algum erro, não criamos outro erro por causa dele.
            if tipo_operando is None:
                return None
            
            # - exige int e produz int
            if expression.operator is UnaryOperator.NEGATE:
                if tipo_operando is TypeName.INT:
                    return set_type(expression, TypeName.INT)

            # ! exige bool e produz bool
            if expression.operator is UnaryOperator.NOT:
                if tipo_operando is TypeName.BOOL:
                    return set_type(expression, TypeName.BOOL)

            # Operando possui o tipo errado.
            diagnostics.append(SemanticDiagnostic(
                SemanticErrorKind.INVALID_UNARY_OPERAND,
                "operando inválido para operador unário",
                expression.span,
            ))
            return None

        # Expressão binária
        if isinstance(expression, BinaryExpr):
            tipo_esquerda = expression_type(expression.left)
            tipo_direita = expression_type(expression.right)

            # Se algum dos lados já possui um erro, não criamos outro erro por causa dele.
            if tipo_esquerda is None or tipo_direita is None:
                return None

            # Operações aritméticas
            if expression.operator in (
                BinaryOperator.ADD,
                BinaryOperator.SUBTRACT,
                BinaryOperator.MULTIPLY,
                BinaryOperator.DIVIDE,
                BinaryOperator.REMAINDER,
            ):
                if (tipo_esquerda is TypeName.INT and tipo_direita is TypeName.INT):
                    return set_type(expression, TypeName.INT)

            # Comparações
            if expression.operator in (
                BinaryOperator.LESS,
                BinaryOperator.LESS_EQUAL,
                BinaryOperator.GREATER,
                BinaryOperator.GREATER_EQUAL,
            ):
                if (tipo_esquerda is TypeName.INT and tipo_direita is TypeName.INT):
                    return set_type(expression, TypeName.BOOL)

            # Igualdade
            if expression.operator in (
                BinaryOperator.EQUAL,
                BinaryOperator.NOT_EQUAL,
            ):
                if (tipo_esquerda is tipo_direita and tipo_esquerda in (TypeName.INT, TypeName.BOOL)):
                    return set_type(expression, TypeName.BOOL)

            # Operações lógicas
            if expression.operator in (
                BinaryOperator.LOGICAL_AND,
                BinaryOperator.LOGICAL_OR,
            ):
                if (tipo_esquerda is TypeName.BOOL and tipo_direita is TypeName.BOOL):
                    return set_type(expression, TypeName.BOOL)

            # Os tipos não são compatíveis com o operador.
            diagnostics.append(SemanticDiagnostic(
                SemanticErrorKind.INVALID_BINARY_OPERANDS,
                "operandos inválidos para operador binário",
                expression.span,
            ))
            return None

        # Chamada de função
        if isinstance(expression, CallExpr):
            # O name_resolver já colocou o símbolo da função dentro do metadata.
            symbol = expression.metadata.get("symbol")

            # Se não existe símbolo, o name_resolver já encontrou um erro, não vamos criar outro erro aqui.
            if not isinstance(symbol, FunctionSymbol):
                return None

            # Primeiro visitamos todos os argumentos.
            argument_types = []

            for argument in expression.arguments:
                argument_types.append(expression_type(argument))

            # Verifica a quantidade de argumentos.
            if len(expression.arguments) != len(symbol.parameter_types):
                diagnostics.append(SemanticDiagnostic(
                    SemanticErrorKind.ARITY_MISMATCH,
                    "quantidade de argumentos incompatível",
                    expression.span,
                ))

            # Verifica os tipos dos argumentos.
            quantidade = min(len(argument_types), len(symbol.parameter_types))
            for i in range(quantidade):
                tipo_argumento = argument_types[i]
                tipo_parametro = symbol.parameter_types[i]

                # Se o argumento já possui um erro, não criamos outro erro causado por esse mesmo problema.
                if tipo_argumento is None:
                    continue

                if tipo_argumento is not tipo_parametro:
                    diagnostics.append(SemanticDiagnostic(
                        SemanticErrorKind.ARGUMENT_TYPE_MISMATCH,
                        "tipo de argumento incompatível",
                        expression.arguments[i].span,
                    ))

            # Função void normalmente não pode ser usada como valor.
            # Porém, quando a chamada aparece como um comando (CallStmt), precisamos permitir a chamada e registrar seu tipo como void.
            if symbol.type is TypeName.VOID:
                if allow_void:
                    return set_type(expression, TypeName.VOID)

                diagnostics.append(SemanticDiagnostic(
                    SemanticErrorKind.VOID_VALUE_USED,
                    "função void não produz valor",
                    expression.span,
                ))
                return None

            # Caso contrário, o tipo da chamada é o retorno da função.
            return set_type(expression, symbol.type)
        
        return None

    def check_block(block: Block, function_return_type: TypeName) -> None:
        # Percorre todos os comandos que existem dentro do bloco.
        for statement in block.statements:
            check_statement(statement, function_return_type)

    def check_statement(statement: Stmt, function_return_type: TypeName) -> None:
        # Bloco aninhado
        if isinstance(statement, Block):
            # Verifica todos os comandos que estão dentro desse bloco.
            check_block(statement, function_return_type)
            return

        # Declaração de variável
        if isinstance(statement, VarDecl):
            # Variáveis não podem ter o tipo void.
            if statement.type is TypeName.VOID:
                diagnostics.append(SemanticDiagnostic(
                    SemanticErrorKind.VOID_VARIABLE,
                    "variável não pode ter tipo void",
                    statement.span,
                ))
                return

            # Se não existe inicializador, não há tipo para comparar.
            if statement.initializer is None:
                return

            # Descobre o tipo da expressão inicializadora.
            tipo_inicializador = expression_type(statement.initializer)

            # Se o inicializador já possui algum erro, não criamos outro erro por causa dele.
            if tipo_inicializador is None:
                return

            # O tipo do inicializador deve ser exatamente o mesmo tipo declarado para a variável.
            if tipo_inicializador is not statement.type:
                diagnostics.append(SemanticDiagnostic(
                    SemanticErrorKind.INITIALIZER_TYPE_MISMATCH,
                    "tipo do inicializador incompatível com a variável",
                    statement.initializer.span,
                ))

        # Atribuição
        if isinstance(statement, Assignment):
            # Descobre o símbolo da variável que está no lado esquerdo.
            symbol = statement.target.metadata.get("symbol")

            # Se não existe símbolo, o name_resolver já encontrou o erro de variável não declarada.
            if not isinstance(symbol, Symbol):
                return

            # Descobre o tipo do valor que será atribuído.
            tipo_valor = expression_type(statement.value)

            # Se o valor já possui algum erro, não criamos outro erro por causa dele.
            if tipo_valor is None:
                return

            # O tipo do valor deve ser exatamente igual ao tipo da variável.
            if tipo_valor is not symbol.type:
                diagnostics.append(SemanticDiagnostic(
                    SemanticErrorKind.ASSIGNMENT_TYPE_MISMATCH,
                    "tipo atribuído incompatível com a variável",
                    statement.value.span,
                ))
            return

        # Chamada de função usada como comando.
        if isinstance(statement, CallStmt):
            # Como a chamada está sendo usada como comando, permitimos que ela seja uma função void.
            expression_type(statement.call, allow_void=True)

            return

        # Comando print
        if isinstance(statement, PrintStmt):
            # Percorre todos os itens que serão impressos.
            for item in statement.items:
                # StringLiteral não é uma Expr.
                # Strings são aceitas diretamente pelo print.
                if isinstance(item, StringLiteral):
                    continue

                # Os outros itens são expressões.
                # expression_type() verifica o tipo e também detecta possíveis erros dentro da expressão.
                expression_type(item)

            return

        # Comando return
        if isinstance(statement, ReturnStmt):
            # Caso seja "return;".
            if statement.value is None:
                # Só é válido dentro de uma função void.
                if function_return_type is not TypeName.VOID:
                    diagnostics.append(SemanticDiagnostic(
                        SemanticErrorKind.RETURN_MISMATCH,
                        "retorno incompatível com o tipo da função",
                        statement.span,
                    ))

                return

            # Caso seja "return expressão;".
            tipo_retorno = expression_type(statement.value)

            # Se a expressão já possui algum erro, não criamos outro erro por causa dela.
            if tipo_retorno is None:
                return

            # A expressão retornada deve ter exatamente o mesmo tipo do retorno da função.
            if tipo_retorno is not function_return_type:
                diagnostics.append(SemanticDiagnostic(
                    SemanticErrorKind.RETURN_MISMATCH,
                    "tipo de retorno incompatível com o tipo da função",
                    statement.value.span,
                ))

            return

        # Estrutura if
        if isinstance(statement, IfStmt):
            # Descobre o tipo da condição.
            tipo_condicao = expression_type(statement.condition)

            # Se a condição possui um tipo válido, verificamos se é bool.
            if tipo_condicao is not None:
                if tipo_condicao is not TypeName.BOOL:
                    diagnostics.append(SemanticDiagnostic(
                        SemanticErrorKind.CONDITION_TYPE_MISMATCH,
                        "a condição deve ser bool",
                        statement.condition.span,
                    ))

            # Verifica os comandos dentro do then.
            check_block(statement.then_block, function_return_type)

            # Se existir else, verifica os comandos dentro dele.
            if statement.else_block is not None:
                check_block(statement.else_block, function_return_type)

            return

        # Estrutura while
        if isinstance(statement, WhileStmt):
            # Descobre o tipo da condição.
            tipo_condicao = expression_type(statement.condition)

            # Se a condição possui um tipo válido, verificamos se é bool.
            if tipo_condicao is not None:
                if tipo_condicao is not TypeName.BOOL:
                    diagnostics.append(SemanticDiagnostic(
                        SemanticErrorKind.CONDITION_TYPE_MISMATCH,
                        "a condição deve ser bool",
                        statement.condition.span,
                    ))

            # Verifica os comandos dentro do corpo do while.
            check_block(statement.body, function_return_type)

            return

    # Percorre todas as funções do programa.
    for function in program.functions:
        # Verifica os parâmetros da função.
        for parameter in function.parameters:
            # Parâmetros não podem ter o tipo void.
            if parameter.type is TypeName.VOID:
                diagnostics.append(SemanticDiagnostic(
                    SemanticErrorKind.VOID_PARAMETER,
                    "parâmetro não pode ter tipo void",
                    parameter.span,
                ))

        # O tipo de retorno da função será passado para os comandos.
        check_block(function.body, function.return_type)

    # Se encontramos algum erro, lançamos todos de uma vez.
    if diagnostics:
        raise SemanticError(diagnostics)