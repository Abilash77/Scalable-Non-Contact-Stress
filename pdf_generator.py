import os
import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_session_pdf(session_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=5,
        alignment=1, # Center
        textColor=colors.HexColor("#1A202C")
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=15,
        alignment=1,
        textColor=colors.HexColor("#4A5568")
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=15,
        spaceAfter=8,
        borderPadding=5,
        backColor=colors.HexColor("#EBF8FF")
    )
    sub_heading = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=10,
        spaceAfter=5
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    alert_style = ParagraphStyle(
        'AlertStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#C53030"),
        backColor=colors.HexColor("#FED7D7"),
        borderPadding=5,
        spaceBefore=5,
        spaceAfter=5
    )
    
    diagram_style = ParagraphStyle(
        'DiagramStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        alignment=1,
        spaceBefore=5,
        spaceAfter=5,
        leading=12
    )

    story = []
    
    # 1. REPORT HEADER
    story.append(Paragraph("Scalable Non-Contact Stress Detection Using Hybrid Multimodal Intelligence", title_style))
    story.append(Paragraph("Multimodal Stress Monitoring — Research Session Report", subtitle_style))
    
    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_info = [
        ["Session ID:", session_data.get('session_id', 'N/A')],
        ["Generated On:", gen_time]
    ]
    t_header = Table(header_info, colWidths=[100, 400])
    t_header.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#718096")),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 15))
    
    # 2. PARTICIPANT INFORMATION
    story.append(Paragraph("PARTICIPANT INFORMATION", section_heading))
    participant = session_data.get('participant', {})
    part_info = [
        ["Name", participant.get('name', 'N/A'), "Session ID", session_data.get('session_id', 'N/A')],
        ["Gender", participant.get('gender', 'N/A'), "Start Time", session_data.get('start_time', 'N/A')],
        ["Age", participant.get('age', 'N/A'), "End Time", session_data.get('end_time', 'N/A')],
        ["", "", "Duration", session_data.get('duration', 'N/A')]
    ]
    t_part = Table(part_info, colWidths=[80, 180, 80, 180])
    t_part.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F7FAFC")),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_part)
    story.append(Spacer(1, 10))
    
    # 3. EXECUTIVE SESSION SUMMARY
    story.append(Paragraph("EXECUTIVE SESSION SUMMARY", section_heading))
    mods = session_data.get('modalities', {})
    active_mods = sum(1 for k, v in mods.items() if v.get('status') not in ['OFFLINE', 'N/A', None])
    
    stress_prob = session_data.get('stress_probability', 0.0)
    non_stress_prob = session_data.get('non_stress_probability', 100.0 - stress_prob)
    
    pred_source = session_data.get('prediction_source', 'N/A')
    source_label = pred_source
    if "Camera" in pred_source or "Heuristic" in pred_source:
        source_label = "CAMERA STRESS ESTIMATE — RESEARCH"
        
    exec_info = [
        ["Final State", session_data.get('final_prediction', 'INSUFFICIENT DATA')],
        ["Stress Score", f"{stress_prob:.2f}%"],
        ["Non-Stress Score", f"{non_stress_prob:.2f}%"],
        ["Available Modalities", f"{active_mods} / 5"],
        ["Session Duration", session_data.get('duration', 'N/A')],
        ["Prediction Source", source_label]
    ]
    t_exec = Table(exec_info, colWidths=[150, 370])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_exec)
    
    if source_label == "CAMERA STRESS ESTIMATE — RESEARCH":
        story.append(Paragraph("<b>Note:</b> Current session prediction is based on heuristic camera estimation, not a clinically validated probability or confidence.", normal_style))
    story.append(Spacer(1, 10))
    
    # 11. CURRENT SESSION RESULTS & 4. FIVE-MODALITY ANALYSIS
    story.append(Paragraph("CURRENT SESSION RESULTS: FIVE-MODALITY ANALYSIS", section_heading))
    
    mod_table_data = [
        ["Modality", "Status", "Feature Dimension", "Data / Sample Status"]
    ]
    
    def format_mod(mod_name, default_dim):
        m = mods.get(mod_name, {})
        status = m.get('status', 'Not available')
        samples = str(m.get('buffer_fill', '0')) + ' valid windows'
        if mod_name == 'facial' and 'face_detection_count' in m:
            samples = f"{m.get('face_detection_count', 0)} faces detected"
        elif mod_name == 'handwriting':
            samples = f"{m.get('buffer_fill', 0)} submissions"
        return [mod_name.capitalize().replace('_', '/'), status, default_dim, samples]

    mod_table_data.append(format_mod('facial', '12D'))
    mod_table_data.append(format_mod('eye_pupil', '5D'))
    mod_table_data.append(format_mod('speech', '169D'))
    mod_table_data.append(format_mod('keyboard', '7D'))
    mod_table_data.append(format_mod('handwriting', '9D'))
    
    t_mod = Table(mod_table_data, colWidths=[100, 100, 120, 200])
    t_mod.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_mod)
    story.append(Spacer(1, 10))
    
    # 5. DATA QUALITY
    story.append(Paragraph("DATA QUALITY", section_heading))
    dq = session_data.get('data_quality', {})
    
    dq_data = [
        ["Camera Availability", "Connected" if dq.get('camera_connected') else "Offline/Not Available"],
        ["Eye/Pupil Tracking", "Ready" if mods.get('eye_pupil', {}).get('features_ready') else "Not available"],
        ["Speech Audio", mods.get('speech', {}).get('status', 'Not available')],
        ["Keyboard Tracking", mods.get('keyboard', {}).get('status', 'Not available')],
        ["Handwriting Submissions", mods.get('handwriting', {}).get('status', 'Not available')]
    ]
    t_dq = Table(dq_data, colWidths=[150, 370])
    t_dq.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_dq)
    story.append(Spacer(1, 15))
    
    # 6. STRESS SCORE TIMELINE
    story.append(Paragraph("STRESS SCORE TIMELINE", section_heading))
    timeline_data = session_data.get('timeline', [])
    if len(timeline_data) >= 2:
        try:
            plt.figure(figsize=(6.5, 3))
            
            # Using actual recorded timestamps and camera stress prob from timeline
            times = [datetime.datetime.strptime(x['timestamp'], "%H:%M:%S") for x in timeline_data if 'camera_stress_prob_pct' in x]
            if times:
                cam_stress = [x['camera_stress_prob_pct'] for x in timeline_data if 'camera_stress_prob_pct' in x]
                plt.plot(times, cam_stress, label='Stress Score (%)', color='#E53E3E', linewidth=2)
                
                plt.title('Live Session Stress Score', fontsize=12)
                plt.xlabel('Time', fontsize=10)
                plt.ylabel('Score (%)', fontsize=10)
                plt.ylim(0, 100)
                plt.grid(True, alpha=0.3)
                
                import matplotlib.dates as mdates
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.gcf().autofmt_xdate()
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                img_buf.seek(0)
                story.append(Image(img_buf, width=420, height=195))
            else:
                story.append(Paragraph("Insufficient timeline data recorded for graph.", normal_style))
        except Exception as e:
            story.append(Paragraph(f"Error generating graph: {str(e)}", normal_style))
    else:
        story.append(Paragraph("Insufficient session duration for a stress timeline graph.", normal_style))
        
    story.append(Spacer(1, 15))
    
    # 7. SESSION EVENT TIMELINE
    story.append(Paragraph("SESSION EVENT TIMELINE", section_heading))
    events = session_data.get('events', [])
    if events:
        evt_table = []
        for e in events:
            evt_table.append([e.get('time', 'N/A'), e.get('event', 'Unknown')])
        t_evt = Table(evt_table, colWidths=[100, 420])
        t_evt.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_evt)
    else:
        # Fallback to basic session events if explicitly recorded events list is empty
        evt_table = [
            [session_data.get('start_time', 'N/A'), "Session started"],
            [gen_time.split(" ")[1], "PDF generated"]
        ]
        t_evt = Table(evt_table, colWidths=[100, 420])
        t_evt.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_evt)
        
    story.append(Spacer(1, 15))
    
    # 8. RESEARCH ARCHITECTURE SUMMARY & 9. MODEL INFORMATION
    arch_elements = []
    arch_elements.append(Paragraph("RESEARCH ARCHITECTURE & MODEL INFO", section_heading))
    
    model_info_data = [
        ["Architecture / Model Name", session_data.get('model_name', 'RA-HMSD')],
        ["Number of Modalities", "5 (Keyboard, Speech, Facial, Eye/Pupil, Handwriting)"],
        ["Temporal Window", "T=10 (10 Timesteps)"],
        ["Input Feature Dimensions", "Audio(169), Face(12), Eye(5), Key(7), Hand(9)"],
        ["Parameter Count", "~214,166"],
        ["Runtime Model Version", "N/A"]
    ]
    t_model = Table(model_info_data, colWidths=[150, 370])
    t_model.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    arch_elements.append(t_model)
    arch_elements.append(Spacer(1, 10))
    
    diagram_text = """
5 NON-CONTACT MODALITIES
(Keyboard, Speech, Facial, Eye/Pupil, Handwriting)
        ↓
FEATURE EXTRACTION
        ↓
MODALITY-SPECIFIC ENCODERS
        ↓
TEMPORAL PROCESSING
        ↓
RELIABILITY ESTIMATION
        ↓
RELIABILITY-AWARE ATTENTION
        ↓
MULTIMODAL FUSION
        ↓
STRESS / NON-STRESS
    """
    arch_elements.append(Paragraph(diagram_text.replace('\n', '<br/>'), diagram_style))
    story.append(KeepTogether(arch_elements))
    
    story.append(Spacer(1, 15))
    
    # 10. RESEARCH RESULTS — CLEAR SEPARATION
    story.append(Paragraph("REFERENCE RESEARCH RESULTS", section_heading))
    story.append(Paragraph("<b>Reported research-paper results — not this participant's session result.</b>", alert_style))
    
    research_res = [
        ["Accuracy", "94.30%"],
        ["Precision", "94.00%"],
        ["Recall", "93.20%"],
        ["F1 Score", "93.60%"],
        ["AUROC", "0.967"],
        ["Calibration Error", "0.036"],
        ["Processing Latency", "38.40 ms/window"]
    ]
    t_res = Table(research_res, colWidths=[150, 370])
    t_res.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 20))
    
    # 12. LIMITATIONS / RESEARCH DISCLAIMER
    story.append(Paragraph("LIMITATIONS & RESEARCH DISCLAIMER", section_heading))
    disclaimer_text = """
    This system is a research prototype for stress-related signal estimation. 
    It is not a medical device and does not diagnose stress disorders, anxiety, depression, 
    personality traits, or other medical or psychological conditions.<br/><br/>
    <b>Individual session scores should not be interpreted as clinical diagnosis.</b>
    """
    story.append(Paragraph(disclaimer_text, normal_style))
    
    doc.build(story)
    return output_path
