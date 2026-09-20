// Intro Page of the entire experiment

import React, { Component } from 'react';

import LabStudyProceedButton from '../Layout-Components/LabStudyProceedButton';


export default class LabStudyIntroPage extends Component {

    constructor(props) {
        super(props);
        this.state = {
            checked: false
        };

        this.toggleCheckBox = this.toggleCheckBox.bind(this);
    }

    componentDidMount() {
        window.scrollTo(0, 0);
    }

    toggleCheckBox() {
        this.state.checked ? this.setState({checked: false}) : this.setState({checked: true})
    }

    render() {
        return(
            <div className="section">
                <div className="content">
                    <h3>Welcome! Thank you very much for supporting this study!</h3><br/><br/>
                    <div style={{textAlign: "left"}}>
                        <h5>Before starting, please read this introductory information:</h5>
                        <p>This study is about the execution of different tasks. On the subsequent pages, you will be asked to work on such tasks and to answer some corresponding questions. The tasks are divided into one practice block and two exercise blocks. In one of those exercise blocks, your performance will be tested. In the other exercise block, your performance will not be tested. Before the start of an exercise block, you will be informed whether your performance will be tested or not.</p>
                        <p>You will receive instructions about what to do in detail. Please read the instructions carefully and work on each task as good as you can.</p>
                        <p><strong>Note: </strong>During the study, please do not minimize this browser window and please do not use the zoom function of this browser. Also, please do not reload this page. You cannot go back once you have clicked on the “continue” button on a study page.</p>
                        <p>If you have questions, please ask the experimenter.</p><br/>
                    </div>
                    <label className="checkbox" style={{display: "flex", alignItems: "center", marginTop: "25px"}}>
                        <input type="checkbox" style={{marginRight: "5px"}} checked={this.state.checked} onChange={this.toggleCheckBox}/>
                        I have read all study information and I am ready to start the study
                    </label>
                    <LabStudyProceedButton onClick={() => this.props.proceedPhase(this.props.phaseName)} disabled={!this.state.checked}/>
                </div>
            </div>
        );
    }

}