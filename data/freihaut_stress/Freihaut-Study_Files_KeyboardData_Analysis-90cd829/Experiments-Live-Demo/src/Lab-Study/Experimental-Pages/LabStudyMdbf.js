// Component that shows the MDBF questionnaire

import React, {Component} from 'react';

import LabStudyMatrixQuestion from '../Layout-Components/LabStudyMatrixQuestion';
import MatrixButtons from '../Layout-Components/LabStudyMatrixButton';
import LabStudyProceedButton from '../Layout-Components/LabStudyProceedButton';


const scaleQuestions = [
    {question: "I am feeling comfortable", id: 'MDBF_wohl'},
    {question: "I am feeling unhappy", id: 'MDBF_unglücklich'},
    {question: "I am feeling dissatisfied", id: 'MDBF_unzufrieden'},
    {question: "I am feeling well", id: 'MDBF_gut'},
    {question: "I am feeling tired", id: 'MDBF_schläfrig'},
    {question: "I am feeling awake", id: 'MDBF_wach'},
    {question: "I am feeling fresh", id: 'MDBF_frisch'},
    {question: "I am feeling exhausted", id: 'MDBF_ermattet'},
    {question: "I am feeling in balance", id: 'MDBF_ausgeglichen'},
    {question: "I am feeling tense", id: 'MDBF_angespannt'},
    {question: "I am feeling nervous", id: 'MDBF_nervös'},
    {question: "I am feeling calm", id: 'MDBF_ruhig'},
    {question: "I am feeling nostalgic", id: 'nostalgia'},
    {question: "I am feeling stressed", id: 'stress'},
];

const scaleAnchors = [
    {anchor: ""},
    {anchor: "strongly disagree"},
    {anchor: "disagree"},
    {anchor: "neutral"},
    {anchor: "agree"},
    {anchor: "strongly agree"}
];


export default class LabStudyMdbf extends Component {

    constructor(props) {
        super(props);

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
        const target = event.target;
        const value = target.value;
        const name = target.name;

        let MDBF = {...this.state.MDBF};
        MDBF[name] = parseInt(value);
        this.setState({MDBF});
    }

    render() {
        return (
            <div className="content" style={{marginTop: "30px"}}>
                <div>
                    <h4>
                        Please select the answer option, which best represents your current feeling.
                    </h4>
                    <hr style={{margin: "0 0", height: "3px"}}/>
                </div>
                <div style={{marginTop: "40px"}}>
                    <table className="table is-hoverable">
                        <tbody>
                        <tr>
                            {scaleAnchors.map((tags, index) => (
                                <th style={{textAlign: "center", width: "70px"}} key={index}>{tags.anchor}</th>
                            ))}
                        </tr>
                        {
                            scaleQuestions.map(question => (
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
        );
    }

}