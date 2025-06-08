import math
import re
import logging
import statistics
from typing import Any, Dict, List, Tuple, Optional
from flask import Flask, request, jsonify, render_template, send_from_directory

# Configuración avanzada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gordis_debug.log"),
        logging.StreamHandler()
    ]
)

# ===== CLASE CONSTANTES MEJORADA =====
class Constantes:
    """Clase para definir constantes matemáticas y científicas."""
    PI = math.pi
    E = math.e
    PHI = (1 + math.sqrt(5)) / 2
    VELOCIDAD_LUZ = 299792458
    GRAVEDAD_ESTANDAR = 9.80665
    PLANCK = 6.62607015e-34
    BOLTZMANN = 1.380649e-23
    AVOGADRO = 6.02214076e23
    CONSTANTE_GRAVITACIONAL = 6.67430e-11
    ACELERACION_GRAVEDAD = 9.80665
    CERO_ABSOLUTO = -273.15

    @classmethod
    def listar(cls) -> Dict[str, float]:
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') and not callable(v)}

# ===== EVALUADOR MATEMÁTICO MEJORADO Y CORREGIDO =====
class EvaluadorMatematico:
    # CORRECCIÓN: Funciones de estadística modificadas para aceptar múltiples argumentos
    FUNCIONES_PERMITIDAS = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'atan2': math.atan2,
        'sqrt': lambda x: math.sqrt(x) if x >= 0 else None,
        'cbrt': lambda x: x ** (1/3),
        'exp': math.exp, 'log': math.log, 'log10': math.log10, 'log2': math.log2,
        'ceil': math.ceil, 'floor': math.floor, 'fabs': math.fabs, 'fmod': math.fmod,
        'factorial': math.factorial, 'sum': sum, 'min': min, 'max': max, 'abs': abs,
        'rad': math.radians, 'deg': math.degrees,
        # Funciones estadísticas corregidas
        'media': lambda *args: statistics.mean(args),
        'mediana': lambda *args: statistics.median(args),
        'moda': lambda *args: statistics.mode(args),
        'varianza': lambda *args: statistics.variance(args),
        'desvest': lambda *args: statistics.stdev(args),
        'pi': Constantes.PI, 'π': Constantes.PI, 'e': Constantes.E,
        'g': Constantes.GRAVEDAD_ESTANDAR, 'c': Constantes.VELOCIDAD_LUZ
    }

    def __init__(self):
        self.namespace = self._crear_namespace()

    def _crear_namespace(self) -> Dict:
        ns = {"__builtins__": {}}
        ns.update(self.FUNCIONES_PERMITIDAS)
        for nombre, valor in Constantes.listar().items():
            ns[nombre.lower()] = valor
        return ns

    def limpiar_expresion(self, expr: str) -> str:
        expr = expr.strip().lower()
        
        # CORRECCIÓN: Manejo de factorial (5! -> factorial(5))
        expr = re.sub(r'(\d+)\s*!', r'factorial(\1)', expr)
        
        # Reemplazos de texto a expresiones matemáticas
        sustituciones = {
            '^': '**', '×': '*', '÷': '/', '²': '**2', '³': '**3',
            '√': 'sqrt(', '∛': 'cbrt(', 'sen': 'sin', 
            'logaritmo natural': 'log', 'ln': 'log',
            'raiz cuadrada': 'sqrt', 'raiz cubica': 'cbrt',
            'valor absoluto': 'abs', 'grados a radianes': 'rad', 
            'radianes a grados': 'deg', 'π': 'pi',
            'pi': 'pi', 'número áureo': 'phi',
            'velocidad de la luz': 'c', 'gravedad': 'g',
            'constante de planck': 'planck', 'número de avogadro': 'avogadro'
        }
        
        for original, reemplazo in sustituciones.items():
            expr = expr.replace(original, reemplazo)
        
        # Manejo especial para funciones trigonométricas
        expr = re.sub(r'(sin|cos|tan)\(\s*π\s*/\s*(\d+)\s*\)', r'\1(pi/\2)', expr)
        expr = re.sub(r'(sin|cos|tan)\(\s*(\d+)\s*°\s*\)', r'\1(rad(\2))', expr)
        expr = re.sub(r'(sin|cos|tan)\(\s*(\d+)\s*grados\s*\)', r'\1(rad(\2))', expr)
        
        # Convertir grados a radianes
        expr = re.sub(r'(\d+)\s*°', r'rad(\1)', expr)
        expr = re.sub(r'(\d+)\s*grados', r'rad(\1)', expr)
        
        # Manejar multiplicación implícita
        expr = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', expr)
        expr = re.sub(r'(\d)\s+(\d)', r'\1\2', expr)
        expr = re.sub(r'\s+', '', expr)
        
        # Manejar expresiones como "2pi" -> "2*pi"
        expr = re.sub(r'(\d)(pi|e|phi|c|g|planck)', r'\1*\2', expr)
        
        # Asegurar paréntesis balanceados
        if expr.count('(') != expr.count(')'):
            expr = self._balancear_parentesis(expr)
        
        return expr

    def _balancear_parentesis(self, expr: str) -> str:
        """Intenta balancear paréntesis automáticamente"""
        abiertos = expr.count('(')
        cerrados = expr.count(')')
        
        if abiertos > cerrados:
            expr += ')' * (abiertos - cerrados)
        else:
            expr = '(' * (cerrados - abiertos) + expr
            
        return expr

    def evaluar(self, expr: str) -> Tuple[Any, bool, str]:
        try:
            expr_limpia = self.limpiar_expresion(expr)
            if not expr_limpia or expr_limpia == "()":
                return None, False, "❌ Expresión no válida"
                
            # Evaluar la expresión
            resultado = eval(expr_limpia, {"__builtins__": {}}, self.namespace)
            
            if resultado is None:
                return None, False, "❌ Operación no soportada"
                
            if isinstance(resultado, complex):
                return None, False, "❌ No se soportan números complejos"
                
            # Redondear para evitar errores de punto flotante
            if isinstance(resultado, float):
                if abs(resultado) < 1e-10:
                    resultado = 0.0
                elif abs(resultado - round(resultado)) < 1e-10:
                    resultado = round(resultado)
            
            return resultado, True, "✅ Cálculo exitoso"
        except ZeroDivisionError:
            return None, False, "❌ División por cero"
        except (ValueError, TypeError) as e:
            return None, False, f"❌ Error: {str(e)}"
        except SyntaxError:
            return None, False, "❌ Sintaxis incorrecta"
        except NameError as e:
            return None, False, f"❌ Variable no definida: {str(e)}"
        except OverflowError:
            return None, False, "❌ Número demasiado grande"
        except Exception as e:
            logging.error(f"Error en evaluación: {str(e)}")
            return None, False, f"❌ Error interno: {type(e).__name__}"

# ===== GORDIS ASISTENTE MEJORADO =====
class Gordis:
    def __init__(self):
        self.evaluador = EvaluadorMatematico()
        self.historial = []

    def procesar_consulta(self, consulta: str) -> Dict[str, Any]:
        consulta = consulta.strip().lower()
        
        # Manejo de mensajes predefinidos
        mensajes_predefinidos = {
            'hola': "👋 ¡Hola! Soy Gordis, tu asistente matemático. ¿En qué puedo ayudarte?",
            'gracias': "🙏 ¡De nada! Siempre a tu servicio",
            'adiós': "👋 ¡Hasta luego! Si necesitas más ayuda, aquí estaré",
            'ayuda': "💡 Puedes preguntarme operaciones como: '2+2', 'sqrt(16)', 'media(5,10,15)', 'sin(45 grados)', o 'convertir 100 km a m'",
            'quién eres': "🤖 Soy Gordis, tu asistente matemático personal. Estoy aquí para ayudarte con cálculos y conversiones científicas.",
            'qué puedes hacer': "🧮 Puedo ayudarte con: cálculos matemáticos, trigonometría, estadística, conversiones de unidades, física básica y más. ¡Pruébame!",
            '': "👋 ¡Hola! ¿En qué puedo ayudarte hoy?"
        }
        
        if consulta in mensajes_predefinidos:
            return self._construir_respuesta(
                mensaje=mensajes_predefinidos[consulta],
                categoria="general"
            )
        
        # Manejo de conversiones de unidades
        conversion_match = re.match(
            r'convertir\s+([\d.]+)\s*([a-z°]+)\s+a\s+([a-z°]+)', 
            consulta, re.IGNORECASE
        )
        if conversion_match:
            return self._manejar_conversion(*conversion_match.groups())
        
        # Manejo de expresiones con variables (ej: radio=5)
        if 'donde' in consulta or 'con' in consulta:
            return self._manejar_variables(consulta)
        
        # Manejo de expresiones matemáticas
        resultado, exito, mensaje = self.evaluador.evaluar(consulta)
        
        if exito:
            return self._construir_respuesta(
                resultado=resultado,
                mensaje=mensaje,
                explicacion=f"Expresión evaluada: {consulta}",
                codigo_generado=f"resultado = {self.evaluador.limpiar_expresion(consulta)}",
                categoria="matemáticas"
            )
        
        return self._construir_respuesta(
            mensaje=mensaje or "🤔 No entendí tu consulta. Intenta con algo como: '2+3*5' o 'sqrt(9)'",
            exito=False,
            categoria="error"
        )

    def _construir_respuesta(
        self,
        resultado: Optional[float] = None,
        mensaje: str = "",
        exito: bool = True,
        explicacion: str = "",
        codigo_generado: str = "",
        categoria: str = ""
    ) -> Dict[str, Any]:
        return {
            "resultado": resultado,
            "exito": exito,
            "mensaje": mensaje,
            "modo": "TXT" if resultado is None else "GN",
            "categoria": categoria,
            "explicacion": explicacion,
            "codigo_generado": codigo_generado
        }

    def _manejar_conversion(self, valor: str, unidad_origen: str, unidad_destino: str) -> Dict[str, Any]:
        try:
            valor_num = float(valor)
            factores = {
                ('km', 'm'): 1000,
                ('m', 'km'): 0.001,
                ('cm', 'm'): 0.01,
                ('m', 'cm'): 100,
                ('mm', 'm'): 0.001,
                ('m', 'mm'): 1000,
                ('kg', 'g'): 1000,
                ('g', 'kg'): 0.001,
                ('lb', 'kg'): 0.453592,
                ('kg', 'lb'): 2.20462,
                ('c', 'f'): lambda x: (x * 9/5) + 32,
                ('f', 'c'): lambda x: (x - 32) * 5/9,
                ('c', 'k'): lambda x: x + 273.15,
                ('k', 'c'): lambda x: x - 273.15,
                ('°c', 'f'): lambda x: (x * 9/5) + 32,
                ('°f', 'c'): lambda x: (x - 32) * 5/9,
                ('°c', 'k'): lambda x: x + 273.15,
                ('k', '°c'): lambda x: x - 273.15,
                ('c', '°f'): lambda x: (x * 9/5) + 32,
                ('f', '°c'): lambda x: (x - 32) * 5/9,
                ('c', 'k'): lambda x: x + 273.15,
                ('k', 'c'): lambda x: x - 273.15,
                ('°c', '°f'): lambda x: (x * 9/5) + 32,
                ('°f', '°c'): lambda x: (x - 32) * 5/9,
                ('°c', 'k'): lambda x: x + 273.15,
                ('k', '°c'): lambda x: x - 273.15,
                ('°f', 'k'): lambda x: (x - 32) * 5/9 + 273.15,
                ('k', '°f'): lambda x: (x - 273.15) * 9/5 + 32,
                ('h', 'min'): lambda x: x * 60,
                ('min', 'h'): lambda x: x / 60,
                ('s', 'min'): lambda x: x / 60,
                ('min', 's'): lambda x: x * 60,
                ('h', 's'): lambda x: x * 3600,
                ('s', 'h'): lambda x: x / 3600,
            }
            
            # Normalizar unidades
            unidad_origen_clean = unidad_origen.lower().replace('°', '').replace('º', '')
            unidad_destino_clean = unidad_destino.lower().replace('°', '').replace('º', '')
            key = (unidad_origen_clean, unidad_destino_clean)
            
            # Buscar conversión directa
            if key in factores:
                factor = factores[key]
                resultado = factor(valor_num) if callable(factor) else valor_num * factor
                return self._construir_respuesta(
                    resultado=resultado,
                    mensaje=f"✅ Conversión: {valor} {unidad_origen} = {resultado:.6g} {unidad_destino}",
                    explicacion=f"Conversión de {unidad_origen} a {unidad_destino}",
                    codigo_generado=f"resultado = {valor} * {factor}" if not callable(factor) else f"resultado = ({valor} * 9/5) + 32" if key == ('c','f') else f"resultado = ({valor} - 32) * 5/9" if key == ('f','c') else f"resultado = {valor} + 273.15" if key == ('c','k') else f"resultado = {valor} - 273.15",
                    categoria="conversión"
                )
            
            # Buscar conversión inversa
            key_inverso = (unidad_destino_clean, unidad_origen_clean)
            if key_inverso in factores:
                factor = factores[key_inverso]
                resultado = valor_num / factor if not callable(factor) else None
                if resultado is not None:
                    return self._construir_respuesta(
                        resultado=resultado,
                        mensaje=f"✅ Conversión: {valor} {unidad_origen} = {resultado:.6g} {unidad_destino}",
                        explicacion=f"Conversión inversa de {unidad_origen} a {unidad_destino}",
                        codigo_generado=f"resultado = {valor} / {factor}",
                        categoria="conversión"
                    )
            
            return self._construir_respuesta(
                mensaje=f"❌ Conversión no soportada: {unidad_origen} a {unidad_destino}",
                exito=False,
                categoria="error"
            )
        except Exception as e:
            logging.error(f"Error en conversión: {str(e)}")
            return self._construir_respuesta(
                mensaje="❌ Error en conversión de unidades",
                exito=False,
                categoria="error"
            )
    
    def _manejar_variables(self, consulta_base: str) -> Dict[str, Any]:
        try:
            # Dividir la expresión y las variables
            if 'donde' in consulta_base:
                expr_principal, variables_str = consulta_base.split('donde', 1)
            else:
                expr_principal, variables_str = consulta_base.split('con', 1)
            
            expr_principal = expr_principal.strip()
            variables = {}
            
            # Extraer variables
            for var_def in variables_str.split(','):
                var_def = var_def.strip()
                if '=' in var_def:
                    nombre, valor = var_def.split('=', 1)
                    nombre = nombre.strip()
                    valor = valor.strip()
                    try:
                        variables[nombre] = float(valor)
                    except ValueError:
                        return self._construir_respuesta(
                            mensaje=f"❌ Valor inválido para variable '{nombre}': '{valor}'",
                            exito=False,
                            categoria="error"
                        )
            
            # Evaluar la expresión principal con las variables
            resultado, exito, mensaje = self.evaluador.evaluar(expr_principal)
            if not exito:
                return self._construir_respuesta(
                    mensaje=mensaje,
                    exito=False,
                    categoria="error"
                )
            
            # Construir código generado
            codigo = f"# Variables:\n"
            for nombre, valor in variables.items():
                codigo += f"{nombre} = {valor}\n"
            codigo += f"resultado = {self.evaluador.limpiar_expresion(expr_principal)}"
            
            return self._construir_respuesta(
                resultado=resultado,
                mensaje=mensaje,
                explicacion=f"Expresión evaluada: {expr_principal} con variables: {variables}",
                codigo_generado=codigo,
                categoria="matemáticas"
            )
            
        except Exception as e:
            logging.error(f"Error con variables: {str(e)}")
            return self._construir_respuesta(
                mensaje="❌ Error al procesar variables en la expresión",
                exito=False,
                categoria="error"
            )

gordis = Gordis()

# ===== FLASK APP MEJORADA =====
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/img/<path:filename>')
def serve_img(filename):
    return send_from_directory('static/img', filename)

@app.route('/ask', methods=['POST'])
def ask_gordis():
    data = request.get_json(force=True)
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({
            "resultado": None,
            "exito": False,
            "mensaje": "❌ Consulta vacía",
            "modo": "N/A",
            "categoria": "error",
            "explicacion": "",
            "codigo_generado": ""
        }), 400
    
    try:
        response_data = gordis.procesar_consulta(query)
        return jsonify(response_data)
    except Exception as e:
        logging.exception("Error grave en /ask:")
        return jsonify({
            "resultado": None,
            "exito": False,
            "mensaje": f"🔥 Error interno: {type(e).__name__}",
            "modo": "N/A",
            "categoria": "error",
            "explicacion": "",
            "codigo_generado": ""
        }), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
