// Component that shows the BFI neuroticism questionnaire used in the study


import React, {Component} from 'react';

import LabStudyMatrixQuestion from '../Layout-Components/LabStudyMatrixQuestion';
import MatrixButtons from '../Layout-Components/LabStudyMatrixButton';
import LabStudyProceedButton from '../Layout-Components/LabStudyProceedButton';

const scaleQuestions = [
    {question: "I am relaxed, I handle stress well.", id: 'NE1R'},
    {question: "I stay optimistic after experiencing a setback.", id: 'NE2R'},
    {question: "I am moody, I have up and down mood swings.", id: 'NE3'},
    {question: "I can be tense.", id: 'NE4'},
    {question: "I feel secure, comfortable with myself.", id: 'NE5R'},
    {question: "I am emotionally stable, not easily upset.", id: 'NE6R'},
    {question: "I worry a lot.", id: 'NE7'},
    {question: "I often feel sad.", id: 'NE8'},
    {question: "I keep my emotions under control.", id: 'NE9R'},
    {question: "I rarely feel anxious or afraid.", id: 'NE10R'},
    {question: "I tend to feel depressed.", id: 'NE11'},
    {question: "I am temperamental, I get emotional easily.", id: 'NE12'},
];

const scaleAnchors = [
    {anchor: ""},
    {anchor: "strongly disagree"},
    {anchor: "disagree"},
    {anchor: "neutral"},
    {anchor: "agree"},
    {anchor: "strongly agree"},
];


export default class LabStudyBfiNeuroticism extends Component {

    constructor(props) {
        super(props);

        // set each intitial questionnaire value to -99 and save them as a state
        let initialMdbf = {};
        scaleQuestions.forEach(item => {
            initialMdbf[item.id] = -99;
        });

        this.state = {MDBF: initialMdbf};

        this.handleInputChange = this.handleInputChange.bind(this);
        this.isQuestionnaireComplete = this.isQuestionnaireComplete.bind(this);
    }

    componentDidMount() {
        window.scrollTo(0, 0);
    }

    isQuestionnaireComplete() {
        // check if all questions have a value other than -99 --> each question is answered, if yes, enable proceedphase button
        for (let property in this.state.MDBF) {
            if (this.state.MDBF.hasOwnProperty(property)) {
                //console.log(property + " = " + this.state[property]);
                if (this.state.MDBF[property] === -99) {
                    return false;
                }
            }
        }

        return true;
    }

    handleInputChange(event) {
        // Get the name and value of the clicked radio button and save it to the corresponding question state
        const target = event.target;
        const value = target.value;
        const name = target.name;

        let MDBF = {...this.state.MDBF};
        MDBF[name] = parseInt(value);
        this.setState({MDBF});
    }

    render() {
        return (
            <div className="section">
                <div className="content" style={{}}>
                    <div>
                        <h4>Please select your agreement to the following statements.</h4>
                        <hr style={{margin: "0 0", height: "3px"}}/>
                    </div>
                    <div style={{marginTop: "40px"}}>
                        <table className="table is-hoverable">
                            <tbody>
                            <tr>
                                {scaleAnchors.map((tags, index) => (
                                    <th style={{textAlign: "center"}} key={index}>{tags.anchor}</th>
                                ))}
                            </tr>
                            {scaleQuestions.map(question => (
                                <tr key={question.question}>
                                    <LabStudyMatrixQuestion question={question.question}/>
                                    {scaleAnchors.slice(1).map((tags, index) => (
                                        <MatrixButtons
                                            key={index}
                                            name={question.id}
                                            value={index}
                                            onChange={this.handleInputChange}
                                        />
                                    ))}
                                </tr>
                            ))
                            }
                            </tbody>
                        </table>
                    </div>
                    <LabStudyProceedButton disabled={!this.isQuestionnaireComplete()}
                                           onClick={() => this.props.proceedPhase(this.props.phaseName, this.state.MDBF)}/>
                </div>
            </div>
        );
    }

}