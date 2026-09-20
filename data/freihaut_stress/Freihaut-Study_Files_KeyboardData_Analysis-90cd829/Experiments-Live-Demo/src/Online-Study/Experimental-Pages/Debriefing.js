import React, { Component } from 'react';

export default class Debriefing extends Component {

    componentDidMount() {
        window.scrollTo(0, 0);
    }

    end(){

        const data = {
            countTaskTotalAnswer: this.props.ans,
            countTaskTotalSolution: this.props.solution
        };

        this.props.proceedPhase(this.props.name, data)
    }

    render() {
        return(
            <div>
                <div className="section">
                    <div className="content" style={{width: "900px"}}>

                        <div style={{textAlign : "left"}}>
                            <h5>
                                Thank you for your answers!
                            </h5>
                            <h5>
                                You counted a total of {this.props.ans} squares and there were a total of {this.props.solution} squares.
                            </h5>
                            <br/>
                            <p>
                                In this study, it was not important whether the squares you counted matched the number of squares shown. It therefore was not a test in a classical sense. The aim of this study was to find out whether the use of a computer mouse and keyboard is influenced by stress.
                                By counting squares {this.props.condition === 0 ? " and announcing a performance test" : ""}, stress should be caused in a part of the participants. We ask you for your understanding if this study was stressfull for you. If you are still feeling tense at the moment, please take some time for yourself to relax.
                            </p>
                            <p>
                                If you have any comments or questions about this study or the research project, please feel free to use the contact below. Thank you for your participation and assistance!
                            </p>
                            <br/>
                            <h5>
                                Contact for queries or comments:
                            </h5>
                            <p>
                                Paul Freihaut, M. Sc. <br/>
                                University of Freiburg <br/>
                                Department of Psychology<br/>
                                Occupational and Consumer Psychology<br/>
                                Engelbergerstraße 41 <br/>
                                D-79085 Freiburg <br/>
                                E-Mail: <a href={"mailto:paul.freihaut@psychologie.uni-freiburg.de"}>paul.freihaut@psychologie.uni-freiburg.de</a>
                            </p>
                            <div>
                            </div>
                            <div style={{marginTop: "50px", display: "flex", alignItems: "center", justifyContent: "center"}} className="control">
                                <button className="button is-link" onClick={() => this.end()}>Finish study</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

}