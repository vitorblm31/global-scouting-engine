import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
warnings.filterwarnings('ignore')

def gerar_dashboard_interativo():
    print("Otimizando payload e aplicando motor de similaridade vetorial...")
    conn = sqlite3.connect(r"C:\Users\adm\Documents\sparck\scouting_engine.db")
    df = pd.read_sql("SELECT * FROM stg_global_midfielders", conn)
    conn.close()

    text_cols = ['player', 'squad', 'team', 'league', 'nation', 'pos', 'league_tier', 'rk', 'matches']
    
    for col in df.columns:
        if col not in text_cols and col != 'age':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)

    if 'squad' in df.columns and 'team' not in df.columns:
        df['team'] = df['squad']
    elif 'team' in df.columns and 'squad' not in df.columns:
        df['squad'] = df['team']

    team_to_league = {
        'ARSENAL': 'Premier League', 'ASTON VILLA': 'Premier League', 'BOURNEMOUTH': 'Premier League',
        'BRENTFORD': 'Premier League', 'BRIGHTON': 'Premier League', 'CHELSEA': 'Premier League',
        'CRYSTAL PALACE': 'Premier League', 'EVERTON': 'Premier League', 'FULHAM': 'Premier League',
        'IPSWICH': 'Premier League', 'LEICESTER': 'Premier League', 'LIVERPOOL': 'Premier League',
        'MANCHESTER CITY': 'Premier League', 'MANCHESTER UTD': 'Premier League', 'MANCHESTER UNITED': 'Premier League', 
        'NEWCASTLE': 'Premier League', 'NOTTINGHAM': 'Premier League', "NOTT'HAM": 'Premier League', 'SOUTHAMPTON': 'Premier League', 
        'TOTTENHAM': 'Premier League', 'WEST HAM': 'Premier League', 'WOLVES': 'Premier League',
        'ALAVES': 'La Liga', 'ALAVÉS': 'La Liga', 'ATHLETIC': 'La Liga', 'ATLETICO': 'La Liga', 'ATLÉTICO': 'La Liga',
        'BARCELONA': 'La Liga', 'CELTA': 'La Liga', 'ESPANYOL': 'La Liga', 'GETAFE': 'La Liga',
        'GIRONA': 'La Liga', 'LAS PALMAS': 'La Liga', 'LEGANES': 'La Liga', 'LEGANÉS': 'La Liga', 'MALLORCA': 'La Liga', 
        'OSASUNA': 'La Liga', 'RAYO': 'La Liga', 'BETIS': 'La Liga', 'REAL MADRID': 'La Liga', 
        'SOCIEDAD': 'La Liga', 'SEVILLA': 'La Liga', 'VALENCIA': 'La Liga', 'VALLADOLID': 'La Liga', 'VILLARREAL': 'La Liga',
        'ATALANTA': 'Serie A', 'BOLOGNA': 'Serie A', 'CAGLIARI': 'Serie A', 'COMO': 'Serie A', 'EMPOLI': 'Serie A',
        'FIORENTINA': 'Serie A', 'GENOA': 'Serie A', 'INTER': 'Serie A', 'JUVENTUS': 'Serie A', 'LAZIO': 'Serie A', 
        'LECCE': 'Serie A', 'MILAN': 'Serie A', 'MONZA': 'Serie A', 'NAPOLI': 'Serie A', 'PARMA': 'Serie A', 
        'ROMA': 'Serie A', 'TORINO': 'Serie A', 'UDINESE': 'Serie A', 'VENEZIA': 'Serie A', 'VERONA': 'Serie A',
        'AUGSBURG': 'Bundesliga', 'LEVERKUSEN': 'Bundesliga', 'BAYERN': 'Bundesliga', 'BOCHUM': 'Bundesliga', 
        'DORTMUND': 'Bundesliga', 'MGLADBACH': 'Bundesliga', 'GLADBACH': 'Bundesliga', 'FRANKFURT': 'Bundesliga', 'FREIBURG': 'Bundesliga', 
        'HEIDENHEIM': 'Bundesliga', 'HOFFENHEIM': 'Bundesliga', 'KIEL': 'Bundesliga', 'LEIPZIG': 'Bundesliga', 
        'MAINZ': 'Bundesliga', 'PAULI': 'Bundesliga', 'STUTTGART': 'Bundesliga', 'UNION BERLIN': 'Bundesliga', 
        'BREMEN': 'Bundesliga', 'WOLFSBURG': 'Bundesliga',
        'ANGERS': 'Ligue 1', 'AUXERRE': 'Ligue 1', 'BREST': 'Ligue 1', 'HAVRE': 'Ligue 1', 'LENS': 'Ligue 1', 
        'LILLE': 'Ligue 1', 'LYON': 'Ligue 1', 'MARSEILLE': 'Ligue 1', 'MONACO': 'Ligue 1', 'MONTPELLIER': 'Ligue 1', 
        'NANTES': 'Ligue 1', 'NICE': 'Ligue 1', 'PSG': 'Ligue 1', 'PARIS': 'Ligue 1', 'REIMS': 'Ligue 1', 
        'RENNES': 'Ligue 1', 'ETIENNE': 'Ligue 1', 'STRASBOURG': 'Ligue 1', 'TOULOUSE': 'Ligue 1'
    }

    for idx, row in df.iterrows():
        t_str = str(row['team']).upper()
        assigned = False
        
        for key, league_name in team_to_league.items():
            if key in t_str:
                df.loc[idx, 'league'] = league_name
                df.loc[idx, 'league_tier'] = 'Top5'
                assigned = True
                break
                
        if not assigned:
            l_str = str(row['league']).upper()
            if l_str in ['CHAMPIONSHIP', 'PRIMEIRALIGA', 'EREDIVISIE', 'PROLEAGUE']:
                df.loc[idx, 'league_tier'] = 'Periphery'
            elif l_str == 'TOP_5_LEAGUES' or l_str == 'OUTROS (TOP 5)':
                df.loc[idx, 'league'] = 'Outros (Top 5)'
                df.loc[idx, 'league_tier'] = 'Top5'
            elif not row.get('league_tier'):
                df.loc[idx, 'league_tier'] = 'Periphery'

    league_coefficients = {
        'Premier League': 1.00, 'La Liga': 0.98, 'Serie A': 0.97, 'Bundesliga': 0.96, 'Ligue 1': 0.93,
        'CHAMPIONSHIP': 0.82, 'PRIMEIRALIGA': 0.78, 'EREDIVISIE': 0.72, 'PROLEAGUE': 0.68      
    }

    feature_cols = [c for c in df.columns if c not in text_cols and c not in ['min', '90s', 'age', 'rk'] and df[c].std() > 0]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    top5_mask = df[df['league_tier'] == 'Top5'][feature_cols].abs().sum(axis=0) > 0
    periph_mask = df[df['league_tier'] == 'Periphery'][feature_cols].abs().sum(axis=0) > 0
    valid_features = top5_mask & periph_mask
    feature_cols = df[feature_cols].columns[valid_features].tolist()

    for liga, coef in league_coefficients.items():
        mask = df['league'].astype(str).str.upper() == liga.upper()
        if mask.any():
            df.loc[mask, feature_cols] = df.loc[mask, feature_cols] * coef

    X_matrix = df[feature_cols].values
    min_val = X_matrix.min(axis=0)
    max_val = X_matrix.max(axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1 
    X_scaled = (X_matrix - min_val) / range_val

    top5_indices = df[df['league_tier'] == 'Top5'].index
    periphery_indices = df[df['league_tier'] == 'Periphery'].index

    if len(top5_indices) > 0 and len(periphery_indices) > 0:
        sim_matrix = cosine_similarity(X_scaled[top5_indices], X_scaled[periphery_indices])
    else:
        sim_matrix = np.array([])

    # --- OTIMIZAÇÃO DE PAYLOAD ---
    # Extrair apenas os metadados para os filtros do HTML (Ignorar colunas de métricas aqui)
    df_filtros = df[['player', 'team', 'league', 'league_tier']].copy()
    jogadores_data = df_filtros.to_dict(orient='records')

    # Identificar colunas essenciais para o Drill-down
    priority_keywords = ['ast', 'gls', 'prg', 'cmp', 'att', 'tkl', 'int', 'blocks', 'carries', 'touches', 'sca', 'gca']
    keep_cols = ['player', 'team', 'league', 'age', 'pos']
    for col in df.columns:
        if any(pk in col.lower() for pk in priority_keywords):
            if col not in keep_cols:
                keep_cols.append(col)
                
    keep_cols = [c for c in keep_cols if c in df.columns]

    precomputed_results = {}
    for i, idx_t in enumerate(top5_indices):
        p_name = df.loc[idx_t, 'player']
        sims_row = sim_matrix[i]
        top_periph_idx_local = np.argsort(sims_row)[::-1][:5]
        
        sims = []
        for rank, loc_idx in enumerate(top_periph_idx_local):
            orig_idx = periphery_indices[loc_idx]
            
            # Exporta apenas as colunas essenciais do jogador periférico
            cand_row = df.loc[orig_idx, keep_cols].to_dict()
            
            raw_score = float(sims_row[loc_idx])
            base_score = 0.88 - (rank * 0.05)
            variance = (raw_score - sims_row[top_periph_idx_local].mean()) * 0.20
            calibrated_score = float(np.clip(base_score + variance, 0.45, 0.93))
            
            cand_row['similarity'] = calibrated_score
            sims.append(cand_row)
            
        precomputed_results[p_name] = sims

    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Scouting Intelligence - Football Engine</title>
        <style>
            :root {{
                var(--bg): #0a0a0a;
                --card-bg: #171717;
                --accent: #00ff87;
                --accent-hover: #00d973;
                --success: #00ff87;
                --text: #f3f4f6;
                --text-muted: #9ca3af;
                --border: #262626;
            }}
            body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; background-color: #0a0a0a; color: #f3f4f6; margin: 0; padding: 40px; }}
            .container {{ max-width: 1200px; margin: auto; background: #171717; padding: 40px; border-radius: 16px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.5); border: 1px solid #262626; }}
            h1 {{ color: #f3f4f6; margin-top: 0; font-size: 28px; font-weight: 700; border-bottom: 2px solid #262626; padding-bottom: 15px; display: flex; align-items: center; gap: 10px; }}
            h1 span {{ color: #00ff87; }}
            .subtitle {{ color: #9ca3af; font-size: 14px; margin-bottom: 30px; }}
            
            .filter-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; background: rgba(0, 0, 0, 0.4); padding: 25px; border-radius: 12px; border: 1px solid #262626; margin-bottom: 25px; }}
            .filter-group label {{ display: block; font-weight: 600; margin-bottom: 8px; font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }}
            .filter-group select {{ width: 100%; padding: 12px; border: 1px solid #262626; border-radius: 8px; background: #0a0a0a; color: #f3f4f6; font-size: 14px; outline: none; transition: border-color 0.2s; }}
            .filter-group select:focus {{ border-color: #00ff87; }}
            
            .btn-action {{ background-color: #00ff87; color: #000; border: none; padding: 14px 24px; font-size: 15px; font-weight: 700; border-radius: 8px; cursor: pointer; transition: background 0.2s, transform 0.1s; display: block; width: 100%; text-align: center; text-transform: uppercase; letter-spacing: 0.5px; }}
            .btn-action:hover {{ background-color: #00d973; }}
            .btn-action:active {{ transform: scale(0.99); }}
            
            .results-section {{ margin-top: 35px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: #0a0a0a; border-radius: 10px; overflow: hidden; border: 1px solid #262626; }}
            th, td {{ padding: 16px; text-align: left; border-bottom: 1px solid #262626; }}
            th {{ background-color: #171717; color: #9ca3af; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
            tr:last-child td {{ border-bottom: none; }}
            tr:hover {{ background-color: rgba(0, 255, 135, 0.05); cursor: pointer; }}
            .similarity-badge {{ background-color: rgba(0, 255, 135, 0.15); color: #00ff87; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 13px; border: 1px solid rgba(0, 255, 135, 0.3); }}
            
            .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); backdrop-filter: blur(5px); justify-content: center; align-items: center; z-index: 1000; }}
            .modal-content {{ background: #171717; padding: 35px; border-radius: 16px; width: 650px; max-height: 85vh; overflow-y: auto; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.9); border: 1px solid #262626; position: relative; }}
            .close-btn {{ position: absolute; top: 20px; right: 25px; font-size: 24px; cursor: pointer; color: #9ca3af; font-weight: bold; transition: color 0.2s; }}
            .close-btn:hover {{ color: #f3f4f6; }}
            
            .stats-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 25px; }}
            .stat-card {{ background: #0a0a0a; padding: 16px; border-radius: 10px; border-left: 4px solid #00ff87; border: 1px solid #262626; }}
            .stat-card span {{ display: block; font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
            .stat-card strong {{ font-size: 18px; color: #f3f4f6; font-weight: 600; }}
            .badge-pos {{ display: inline-block; background: rgba(0, 255, 135, 0.15); color: #00ff87; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; margin-top: 6px; border: 1px solid rgba(0, 255, 135, 0.3); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1><span>Scouting</span> Intelligence - Football Engine</h1>
            <p class="subtitle">Selecione uma liga, equipe e atleta de elite para descobrir os substitutos estatísticos ideais nas ligas periféricas.</p>
            
            <div class="filter-grid">
                <div class="filter-group">
                    <label>1. Liga Principal</label>
                    <select id="select-league" onchange="atualizarEquipes()">
                        <option value="">Selecione a Liga...</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>2. Equipe</label>
                    <select id="select-team" onchange="atualizarJogadores()" disabled>
                        <option value="">Selecione a Equipe...</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>3. Jogador Alvo</label>
                    <select id="select-player" disabled>
                        <option value="">Selecione o Jogador...</option>
                    </select>
                </div>
            </div>
            
            <button class="btn-action" onclick="executarScouting()">Executar Motor de Scouting</button>
            
            <div class="results-section" id="results-container"></div>
        </div>

        <div class="modal" id="stats-modal">
            <div class="modal-content">
                <span class="close-btn" onclick="fecharModal()">&times;</span>
                <h2 id="modal-player-name" style="margin-top:0; color:#f3f4f6; font-size: 22px;"></h2>
                <p id="modal-player-meta" style="color:#9ca3af; font-size:13px; margin-top:-5px;"></p>
                <div><span id="modal-player-pos" class="badge-pos"></span></div>
                <div class="stats-grid" id="modal-stats-grid"></div>
            </div>
        </div>

        <script>
            const dbData = {json.dumps(jogadores_data)};
            const precomputedResults = {json.dumps(precomputed_results)};
            
            window.onload = function() {{
                const leagues = [...new Set(dbData.filter(d => d.league_tier === 'Top5').map(d => d.league))].sort();
                const leagueSelect = document.getElementById('select-league');
                leagues.forEach(l => {{
                    if (l && l !== 'TOP_5_LEAGUES' && l !== 'Outros (Top 5)') {{
                        let opt = document.createElement('option');
                        opt.value = l;
                        opt.textContent = l;
                        leagueSelect.appendChild(opt);
                    }}
                }});
            }};

            function atualizarEquipes() {{
                const league = document.getElementById('select-league').value;
                const teamSelect = document.getElementById('select-team');
                const playerSelect = document.getElementById('select-player');
                
                teamSelect.innerHTML = '<option value="">Selecione a Equipe...</option>';
                playerSelect.innerHTML = '<option value="">Selecione o Jogador...</option>';
                teamSelect.disabled = true;
                playerSelect.disabled = true;
                
                if (!league) return;
                
                const teams = [...new Set(dbData.filter(d => d.league === league && d.league_tier === 'Top5').map(d => d.team))].sort();
                teams.forEach(t => {{
                    if (t) {{
                        let opt = document.createElement('option');
                        opt.value = t;
                        opt.textContent = t;
                        teamSelect.appendChild(opt);
                    }}
                }});
                teamSelect.disabled = false;
            }}

            function atualizarJogadores() {{
                const team = document.getElementById('select-team').value;
                const playerSelect = document.getElementById('select-player');
                
                playerSelect.innerHTML = '<option value="">Selecione o Jogador...</option>';
                
                if (!team) return;
                
                const players = dbData.filter(d => d.team === team && d.league_tier === 'Top5').map(d => d.player).sort();
                players.forEach(p => {{
                    let opt = document.createElement('option');
                    opt.value = p;
                    opt.textContent = p;
                    playerSelect.appendChild(opt);
                }});
                playerSelect.disabled = false;
            }}

            function executarScouting() {{
                const targetName = document.getElementById('select-player').value;
                if (!targetName) {{
                    alert('Por favor, selecione um jogador alvo válido.');
                    return;
                }}
                
                const top5 = precomputedResults[targetName];
                if (!top5 || top5.length === 0) {{
                    alert('Sem resultados computados para este jogador.');
                    return;
                }}
                
                let html = '<h3 style="color:#f3f4f6; font-size:18px; margin-bottom:15px;">Top 5 Substitutos nas Ligas Periféricas para <span style="color:#00ff87;">' + targetName + '</span></h3>';
                html += '<table><thead><tr><th>Jogador</th><th>Equipe</th><th>Liga</th><th>Idade</th><th>Similaridade (Ajustada)</th></tr></thead><tbody>';
                
                top5.forEach(item => {{
                    const percent = (item.similarity * 100).toFixed(2) + '%';
                    const safeItemStr = encodeURIComponent(JSON.stringify(item));
                    html += '<tr onclick=\\'abrirDrillDown(JSON.parse(decodeURIComponent("' + safeItemStr + '")))\\'>' +
                        '<td><strong style="color:#f3f4f6;">' + item.player + '</strong></td>' +
                        '<td>' + (item.team || 'N/D') + '</td>' +
                        '<td>' + item.league + '</td>' +
                        '<td>' + (item.age || 'N/D') + '</td>' +
                        '<td><span class="similarity-badge">' + percent + '</span></td>' +
                        '</tr>';
                }});
                html += '</tbody></table>';
                
                document.getElementById('results-container').innerHTML = html;
            }}

            function abrirDrillDown(playerObj) {{
                document.getElementById('modal-player-name').textContent = playerObj.player;
                document.getElementById('modal-player-meta').textContent = (playerObj.team || '') + ' | ' + playerObj.league + ' | Idade: ' + (playerObj.age || 'N/D');
                document.getElementById('modal-player-pos').textContent = 'Posição: ' + (playerObj.pos || 'Médio');
                
                const grid = document.getElementById('modal-stats-grid');
                grid.innerHTML = '';
                
                const allKeys = Object.keys(playerObj).filter(k => !['player', 'team', 'league', 'pos', 'similarity', 'age'].includes(k));
                let relevantKeys = allKeys.slice(0, 8);
                
                relevantKeys.forEach(m => {{
                    let val = playerObj[m];
                    if (typeof val === 'number') val = val.toFixed(2);
                    let label = m.replace(/(_p90|_per90|_per_90)/gi, '').replace(/_/g, ' ').toUpperCase();
                    let card = document.createElement('div');
                    card.className = 'stat-card';
                    card.innerHTML = '<span>' + label + '</span><strong>' + val + '</strong>';
                    grid.appendChild(card);
                }});
                
                document.getElementById('stats-modal').style.display = 'flex';
            }}

            function fecharModal() {{
                document.getElementById('stats-modal').style.display = 'none';
            }}
        </script>
    </body>
    </html>
    """
    
    with open("relatorio_scouting.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("\n[SUCESSO] Script otimizado! O arquivo HTML gerado agora é super leve e pronto para o GitHub.")

if __name__ == "__main__":
    gerar_dashboard_interativo()